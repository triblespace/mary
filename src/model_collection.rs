//! Native policy collections and compatibility projection for Mary models.
//!
//! A model collection is its typed descriptor handle. New collections use
//! independent direct READ and WRITE policies rooted at their founding signer;
//! an existing descriptor keeps its own policy. Callers never reconstruct a
//! collection from a separate group coordinate. Reads discover the sole
//! matching descriptor in one frozen pile snapshot, admit its exact support, and
//! materialize that same observation. Writes either join that descriptor after
//! checking its WRITE policy or found the deterministic direct-policy default.
//!
//! TribleSpace commit 6b65f278 changed anchored attribute identity. Model piles
//! written before that epoch contain literal ids. This module preserves those
//! facts and adds their canonical anchored-attribute aliases.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use anyhow::{Context, anyhow, bail};
use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace::core::attribute::Attribute;
use triblespace::core::blob::encodings::simplearchive::SimpleArchive;
use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
use triblespace::core::collection::descriptor;
use triblespace::core::collection::simplearchive_union::PreparedCollectionCommit;
use triblespace::core::collection::{
    ACTION_READ, ACTION_WRITE, AdmissionPolicy, Collection, CollectionCommit, CollectionHandle,
    CollectionPolicy, CollectionRead, CollectionRecord, CollectionSnapshotExt, CollectionStoreExt,
    Support,
};
use triblespace::core::inline::encodings::UnknownInline;
use triblespace::core::metadata;
use triblespace::core::repo::pile::PileSnapshot;
use triblespace::core::repo::{BlobStoreGet, SnapshotSource, StoreRead};
use triblespace::core::trible::TribleSet;
use triblespace::prelude::inlineencodings::{F64, U256BE};
use triblespace::prelude::*;

pub const fn mary_model_graph_name() -> &'static str {
    "mary-model-graph"
}

pub const fn mary_model_bundle_name() -> &'static str {
    "mary-model-bundles"
}

pub type ModelCollection = Collection<SimpleArchive>;

fn direct_model_policy(root: VerifyingKey) -> CollectionPolicy {
    CollectionPolicy::new(AdmissionPolicy::direct(root), AdmissionPolicy::direct(root))
}

/// One coherent observation of materialized model facts.
#[derive(Clone, Debug)]
pub struct ModelSnapshot<S> {
    facts: TribleSet,
    support: Support,
    store: S,
}

impl<S> ModelSnapshot<S> {
    pub fn new(facts: TribleSet, support: Support, store: S) -> Self {
        Self {
            facts,
            support,
            store,
        }
    }

    pub fn facts(&self) -> &TribleSet {
        &self.facts
    }

    pub fn support(&self) -> &Support {
        &self.support
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn into_facts(self) -> TribleSet {
        self.facts
    }

    pub fn into_parts(self) -> (TribleSet, Support, S) {
        (self.facts, self.support, self.store)
    }
}

pub type ModelPileSnapshot = ModelSnapshot<PileSnapshot>;

#[derive(Clone, Debug)]
pub struct PreparedModelBundle {
    model_root: Id,
    model_archive_data: triblespace::core::collection::CollectionData,
    prepared: PreparedCollectionCommit,
}

impl PreparedModelBundle {
    pub fn model_root(&self) -> Id {
        self.model_root
    }

    pub fn model_archive_data(&self) -> triblespace::core::collection::CollectionData {
        self.model_archive_data
    }

    pub fn into_prepared_commit(self) -> PreparedCollectionCommit {
        self.prepared
    }
}

pub fn prepare_model_fragment(fragment: Fragment) -> PreparedCollectionCommit {
    PreparedCollectionCommit::from_fragment(fragment)
}

fn named_collections_from_handles(
    store: &PileSnapshot,
    wanted: &str,
    handles: impl IntoIterator<Item = CollectionHandle>,
) -> (Vec<ModelCollection>, Vec<CollectionHandle>) {
    let mut collections = Vec::new();
    let mut retired = Vec::new();
    for handle in handles {
        let Ok(blob) = store.get::<Blob<SimpleArchive>, _>(handle.transmute()) else {
            continue;
        };
        let Ok(facts) = TribleSet::try_from_blob(blob) else {
            continue;
        };
        let Ok(Some(name_handle)) = descriptor::name(&facts) else {
            continue;
        };
        let Ok(name) = store.get::<anybytes::View<str>, _>(name_handle) else {
            continue;
        };
        if &*name != wanted {
            continue;
        }
        // Opening checks the member encoding, not the policy vocabulary.
        // Older descriptors can still name SimpleArchive while their policy
        // links are no longer understood. Keep them out of current discovery
        // so an additive migration can coexist with its original records.
        // Do not filter by admitted support: a current, empty collection is
        // still current, including when this snapshot admits no writer.
        // Ordinary policy admission accepts supported alternatives; the scalar
        // descriptor inspector would incorrectly reject plural policy bindings.
        let has_current_policies = [ACTION_READ, ACTION_WRITE].into_iter().all(|action| {
            descriptor::admission_policies(store, &facts, action, Some(SimpleArchive::id()))
                .next()
                .is_some()
        });
        match (ModelCollection::open(store, handle), has_current_policies) {
            (Ok(collection), true) => collections.push(collection),
            _ => retired.push(handle),
        }
    }
    collections.sort_unstable();
    collections.dedup();
    retired.sort_unstable();
    retired.dedup();
    (collections, retired)
}

fn named_collections_in(
    store: &PileSnapshot,
    wanted: &str,
) -> anyhow::Result<Vec<ModelCollection>> {
    let mut handles = BTreeSet::new();
    for record in store.records().context("read model collection records")? {
        if let CollectionRecord::Commit(commit) =
            record.context("decode model collection record")?
        {
            handles.insert(commit.collection());
        }
    }

    let (collections, retired) = named_collections_from_handles(store, wanted, handles);
    if collections.is_empty() && !retired.is_empty() {
        bail!(
            "found only retired descriptors named '{wanted}' ({retired:?}); \
             run the additive model collection migration"
        );
    }
    Ok(collections)
}

fn sole_named_collection_in(
    store: &PileSnapshot,
    wanted: &'static str,
) -> anyhow::Result<ModelCollection> {
    let collections = named_collections_in(store, wanted)?;
    match collections.as_slice() {
        [collection] => Ok(*collection),
        [] => bail!("no collection named '{wanted}' in this pile"),
        _ => bail!(
            "{} policy collections are named '{wanted}'; select one explicitly: {:?}",
            collections.len(),
            collections,
        ),
    }
}

pub fn model_graph_collections_in(store: &PileSnapshot) -> anyhow::Result<Vec<ModelCollection>> {
    named_collections_in(store, mary_model_graph_name())
}

pub fn model_bundle_collections_in(store: &PileSnapshot) -> anyhow::Result<Vec<ModelCollection>> {
    named_collections_in(store, mary_model_bundle_name())
}

/// Select the sole collection named `name`, or found it under this signer.
///
/// Discovery and the WRITE check share one frozen observation, and neither
/// consults a clock: descriptors are discovered from records and admission is
/// decided from capability proofs. There is no evaluation time for a caller to
/// choose here, so this takes none.
pub(crate) fn collection_or_create(
    pile: &mut Pile,
    signing_key: &SigningKey,
    name: &'static str,
) -> anyhow::Result<ModelCollection> {
    let signer = signing_key.verifying_key();
    let snapshot = pile
        .snapshot()
        .context("freeze model collection selection")?;
    let collections = named_collections_in(&snapshot, name)?;
    match collections.as_slice() {
        [] => pile
            .collection(name, direct_model_policy(signer))
            .map_err(|error| anyhow!("register '{name}' collection: {error}")),
        [collection] => {
            let admitted = collection
                .writer_is_admitted(&snapshot, signer)
                .with_context(|| format!("check WRITE policy for '{name}'"))?;
            if !admitted {
                bail!(
                    "signer {:?} is not admitted by '{name}' collection {:?}",
                    signer.to_bytes(),
                    collection,
                );
            }
            Ok(*collection)
        }
        _ => bail!(
            "{} policy collections are named '{name}'; refusing to choose by accident",
            collections.len(),
        ),
    }
}

pub fn model_graph_collection_or_create(
    pile: &mut Pile,
    signing_key: &SigningKey,
) -> anyhow::Result<ModelCollection> {
    collection_or_create(pile, signing_key, mary_model_graph_name())
}

pub fn model_bundle_collection_or_create(
    pile: &mut Pile,
    signing_key: &SigningKey,
) -> anyhow::Result<ModelCollection> {
    collection_or_create(pile, signing_key, mary_model_bundle_name())
}

pub fn publish_model_fragment(
    pile: &mut Pile,
    signing_key: &SigningKey,
    fragment: Fragment,
) -> anyhow::Result<CollectionCommit> {
    pile.refresh().context("refresh before model publication")?;
    let collection = model_graph_collection_or_create(pile, signing_key)?;
    pile.commit(collection, signing_key, fragment)
        .map_err(|error| anyhow!("publish model fragment: {error}"))
}

pub fn prepare_model_bundle_fragment(
    model_root: Id,
    fragment: Fragment,
) -> anyhow::Result<PreparedModelBundle> {
    if !fragment.facts().iter().any(|fact| fact.e() == &model_root) {
        bail!("model root {model_root} is absent from the candidate fragment");
    }

    let (_, facts, metafacts, blobs) = fragment.into_parts();
    let model_archive: Blob<SimpleArchive> = facts.to_blob();
    let mut token = Fragment::empty();
    token.blobs_mut().union(blobs);
    let source = token.put::<SimpleArchive, _>(model_archive);
    let model_archive_data = inlineencodings::Handle::<SimpleArchive>::to_hash(source);
    token += entity! {
        ExclusiveId::force_ref(&model_root) @ metadata::archive: source
    };
    *token.metafacts_mut() += metafacts;
    debug_assert_eq!(token.facts().len(), 1);

    Ok(PreparedModelBundle {
        model_root,
        model_archive_data,
        prepared: PreparedCollectionCommit::from_fragment(token),
    })
}

pub fn publish_model_bundle_fragment(
    pile: &mut Pile,
    signing_key: &SigningKey,
    model_root: Id,
    fragment: Fragment,
) -> anyhow::Result<CollectionCommit> {
    pile.refresh()
        .context("refresh before model bundle publication")?;
    let collection = model_bundle_collection_or_create(pile, signing_key)?;
    prepare_model_bundle_fragment(model_root, fragment)?
        .into_prepared_commit()
        .stage_for(pile, collection, signing_key)
        .map_err(|error| anyhow!("stage model bundle: {error}"))?
        .finalize()
        .map_err(|error| anyhow!("publish model bundle: {error}"))
}

pub fn snapshot_model_collection_exact<R: StoreRead>(
    store: &R,
    support: &Support,
) -> anyhow::Result<ModelSnapshot<R>> {
    let facts = store
        .collection_exact(support.collection(), support)
        .context("attach exact model support")?
        .view::<TribleSet>()
        .context("materialize exact model support")?;
    Ok(ModelSnapshot::new(facts, support.clone(), store.clone()))
}

pub fn snapshot_model_bundle_collection_exact<R: StoreRead>(
    store: &R,
    support: &Support,
) -> anyhow::Result<ModelSnapshot<R>> {
    snapshot_model_collection_exact(store, support)
}

pub fn local_model_support<R: StoreRead>(
    store: &R,
    collection: ModelCollection,
) -> anyhow::Result<Support> {
    collection
        .admitted(store)
        .context("admit local model collection support")
}

fn open_and_refresh_model_pile(path: &Path) -> anyhow::Result<Pile> {
    let mut pile = Pile::open(path).context("open model pile")?;
    if let Err(error) = pile.refresh() {
        let _ = pile.close();
        return Err(error).context("refresh model pile");
    }
    Ok(pile)
}

pub fn load_model_collection_from_support(
    path: impl AsRef<Path>,
    support: &Support,
) -> anyhow::Result<ModelPileSnapshot> {
    let mut pile = open_and_refresh_model_pile(path.as_ref())?;
    let store = match pile.snapshot().context("freeze model pile") {
        Ok(store) => store,
        Err(error) => {
            let _ = pile.close();
            return Err(error);
        }
    };
    let materialized = snapshot_model_collection_exact(&store, support);
    let close = pile.close().context("close model pile after snapshot");
    match (materialized, close) {
        (Ok(snapshot), Ok(())) => Ok(snapshot),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

pub fn snapshot_model_collection_local_latest(
    pile: &mut Pile,
) -> anyhow::Result<ModelPileSnapshot> {
    let store = pile.snapshot().context("freeze model pile observation")?;
    snapshot_model_collection_in(&store)
}

pub fn snapshot_model_collection_in(store: &PileSnapshot) -> anyhow::Result<ModelPileSnapshot> {
    snapshot_model_collection_named_in(store, mary_model_graph_name())
}

/// The sole model collection of this name, frozen from the local observation.
/// Model versions are roots inside that collection, not separately named
/// collections. Callers holding a descriptor use [`snapshot_model_collection_for`]
/// directly and select the actual model root from its facts.
pub fn snapshot_model_collection_named_local_latest(
    pile: &mut Pile,
    name: &'static str,
) -> anyhow::Result<ModelPileSnapshot> {
    let store = pile.snapshot().context("freeze model pile observation")?;
    snapshot_model_collection_named_in(&store, name)
}

pub fn snapshot_model_collection_named_in(
    store: &PileSnapshot,
    name: &'static str,
) -> anyhow::Result<ModelPileSnapshot> {
    let collection = sole_named_collection_in(store, name)?;
    snapshot_model_collection_for(store, collection)
}

/// Read the resident part of one explicit model collection from a frozen store.
///
/// The collection handle names where to query; root entity IDs name the models.
/// Physical member archives and unrelated observations are not model identity.
/// The reader may be a pile snapshot, an in-memory store, or the bounded store
/// observation supplied to a collection mapping.
pub fn snapshot_model_collection_for<R: StoreRead>(
    store: &R,
    collection: ModelCollection,
) -> anyhow::Result<ModelSnapshot<R>> {
    let observed = store
        .collection(collection)
        .context("attach model collection")?;
    let facts = observed
        .view::<TribleSet>()
        .context("read model collection")?;
    let (store, support, _) = observed
        .into_parts()
        .context("resolve model collection support for snapshot")?;
    Ok(ModelSnapshot::new(facts, support, store))
}

pub fn snapshot_model_bundle_collection_local_latest(
    pile: &mut Pile,
) -> anyhow::Result<ModelPileSnapshot> {
    let store = pile.snapshot().context("freeze model bundle observation")?;
    snapshot_model_bundle_collection_in(&store)
}

pub fn snapshot_model_bundle_collection_in(
    store: &PileSnapshot,
) -> anyhow::Result<ModelPileSnapshot> {
    let collection = sole_named_collection_in(store, mary_model_bundle_name())?;
    snapshot_model_collection_for(store, collection)
}

pub fn load_model_collection_local_latest(
    path: impl AsRef<Path>,
) -> anyhow::Result<ModelPileSnapshot> {
    let mut pile = open_and_refresh_model_pile(path.as_ref())?;
    let snapshot = snapshot_model_collection_local_latest(&mut pile);
    let close = pile.close().context("close model pile after snapshot");
    match (snapshot, close) {
        (Ok(snapshot), Ok(())) => Ok(snapshot),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

/// Number of unique pre-epoch model-graph attributes projected by this module.
///
/// The historical declarations occupied more source sites because `index`,
/// `member`, and `model_name` were shared by the format and tokenizer schemas.
pub const LEGACY_MODEL_ATTRIBUTE_ALIAS_COUNT: usize = 45;

/// One historical-literal to canonical-anchored attribute mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelAttributeAlias {
    /// Stable diagnostic label naming the schema and field.
    pub label: &'static str,
    /// Literal id present in model piles written before the attribute epoch.
    pub historical: Id,
    /// Encoding-aware anchored id used by current runtime declarations.
    pub canonical: Id,
}

/// Per-attribute counts produced while projecting one fact set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelAttributeAliasCounts {
    /// Mapping these counts describe.
    pub alias: ModelAttributeAlias,
    /// Historical facts encountered under this mapping.
    pub historical_facts: usize,
    /// Canonical aliases newly added to the returned union.
    pub aliases_added: usize,
    /// Historical facts whose exact canonical alias was already present.
    pub aliases_already_present: usize,
}

/// Additive projection result and diagnostics.
#[derive(Clone, Debug)]
pub struct ModelAttributeProjection {
    /// The complete input fact set unioned with missing canonical aliases.
    pub facts: TribleSet,
    /// Number of input facts scanned.
    pub input_facts: usize,
    /// Number of facts whose attribute was from the historical model schema.
    pub historical_facts: usize,
    /// Total number of canonical aliases added to [`Self::facts`].
    pub aliases_added: usize,
    /// Exhaustive mapping table with per-attribute counts, including zeroes.
    pub mappings: Vec<ModelAttributeAliasCounts>,
}

// These declarations exist only to name bytes already found in pre-epoch
// piles. `unsafe as` is intentional: unlike the runtime schemas, this side of
// the mapping must denote the historical literal id exactly. No post-epoch
// Inkling or dataset/training attributes belong in this module.
mod historical {
    use crate::f16enc::F16Array;
    use crate::format::{F32Array, U32Array, U64Array};
    use triblespace::prelude::blobencodings::{RawBytes, UTF8String};
    use triblespace::prelude::inlineencodings::{Boolean, F64, GenId, Handle, ShortString, U256BE};
    use triblespace::prelude::*;

    attributes! {
        // Model format and graph, all present before 6b65f278.
        "572B45D52A47608F283D0F778597137A" unsafe as data: Handle<F32Array>;
        "467CCF3FDCCCCE599F6C1B933EACD933" unsafe as data_f16: Handle<F16Array>;
        "D09A91FC3F04C40AE4A42CD6628A9E38" unsafe as shape: Handle<U64Array>;
        "2ADC6462A7F70E230558C5D681E38768" unsafe as data_q4: Handle<U32Array>;
        "23178058559C762BB4B1FEAA36B3566D" unsafe as data_q8: Handle<U32Array>;
        "F9EA2FB90DC094D42A4845B013950032" unsafe as q_scales: Handle<F16Array>;
        "2CC4D16369C4980BCB512937DA204FF5" unsafe as format_marker: GenId;
        "4629D277AD6B52B50DA78DEF63440AF1" unsafe as weight: GenId;
        "18E898172078C843A0351C3D880CC238" unsafe as bias: GenId;
        "52C4A211D2A08BA25C27FFD79FF24C93" unsafe as kind: ShortString;
        "09EA2F7BCF9B0C9714EE39CF269DF2D5" unsafe as safetensor_path: Handle<UTF8String>;
        "33CE12B1B940B13E48D8E5B0ADFD2421" unsafe as index: U256BE;
        "3F46CDE630964D78D62DA32F4A8558C1" unsafe as model_root: GenId;
        "B4B6EC08A0CD70DE63A690168EE78F0F" unsafe as member: GenId;
        "4C1CD1611863E7854C59C7DC706DF77A" unsafe as model_name: Handle<UTF8String>;
        "D20B8E3556C35FF6D18D104C3443D6CF" unsafe as source: Handle<UTF8String>;
        "7AF87320C144AA29C29FE2A5EE7C7EB2" unsafe as quantization: ShortString;

        // Gemma LoRA, also present before 6b65f278. `model_name` is shared
        // with the format schema above and therefore has only one mapping.
        "1A682F45CE40171DD5C6FDB4F086AD69" unsafe as lora_rank: U256BE;
        "198B03AF556B7505CCC9ABD4A1D6E724" unsafe as lora_alpha: F64;
        "B93C4E66F4B9553BF0E8B5DBAD116ECF" unsafe as lora_adapter: GenId;
        "FF8335C187823A267E26B4E33EF157E9" unsafe as lora_projection: ShortString;
        "7CD7F0DC8BDA328735A22DF02B4B8828" unsafe as lora_a: Handle<F32Array>;
        "1F21DAE68652A4D8CAD973400F04124D" unsafe as lora_b: Handle<F32Array>;

        // Tokenizer graph. `piece_bytes` was introduced by d6dcbd3a on
        // 2026-08-05, four days before 6b65f278, so it belongs to this epoch.
        // The shared index/member/model_name attributes are already above.
        "E7014108A8F9512B19E3E8272E8A71F9" unsafe as tokenizer: GenId;
        "E839AA8F549C0D608FB86476A1EF3416" unsafe as vocab: GenId;
        "E229769197BB035A2D6F61BC6A7D44BC" unsafe as merge: GenId;
        "B2553118F4CAAF1D028619956DE7F145" unsafe as added: GenId;
        "53BAF87A0E7F1410F8212B3EDF2A498C" unsafe as normalizer: GenId;
        "6EEBF39CADD11B7CFBB624019AE21585" unsafe as pre_tokenizer: GenId;
        "98EC58B28F4D0BB43965DF7C5FF22713" unsafe as post_processor: GenId;
        "F3AAA4CD8EE04E5592059564A21FE953" unsafe as decoder: GenId;
        "AE7FE29F2F38153F58C542D5CA4A9356" unsafe as piece: Handle<UTF8String>;
        "F0E2E782F7BB62F52B1186DDE0EB5388" unsafe as token_id: U256BE;
        "714AE13F801202EB27C83E3AB2290669" unsafe as piece_bytes: Handle<RawBytes>;
        "5723ECE1FF426C58879B79D5669A7CF1" unsafe as merge_left: Handle<UTF8String>;
        "5C78FEB151F35A2C5D07BEC92E860752" unsafe as merge_right: Handle<UTF8String>;
        "68F1A9E6ED735E7C3ADCCA076AFF1742" unsafe as unk_token: ShortString;
        "11F76A2C0856C16CB030C4327D5A3B93" unsafe as continuing_subword_prefix: ShortString;
        "6FB969E8A3EDD1A657C721DD5A7D42EA" unsafe as end_of_word_suffix: ShortString;
        "DF3F88DBFA2B44A7783169C9640014AF" unsafe as max_input_chars: U256BE;
        "3BCB70478942DB710ED2A4FB023F3457" unsafe as piece_score: F64;
        "EE4C6647619A836326196F0DBF84FA98" unsafe as byte_fallback: Boolean;
        "C8262D5668B8A1F541B3C35D54201BEC" unsafe as pattern: Handle<UTF8String>;
        "3AC7574C07D02D389B4E7AD3B3B084D9" unsafe as replace_content: ShortString;
        "964B4FCF7477E7E4436F0325F89B7CB5" unsafe as behavior: ShortString;
    }
}

/// Return the complete audited mapping used by
/// [`project_legacy_model_attributes`].
///
/// Gemma is feature-gated, so its LoRA attributes are derived here from the
/// same anchors and encodings as `models::gemma::lora::attrs`; the remaining
/// canonical ids come directly from the unconditional runtime declarations.
/// Canonical targets are disjoint from all historical sources, so the table
/// cannot contain an `A -> B -> C` chain that would need a second pass.
pub fn legacy_model_attribute_aliases() -> [ModelAttributeAlias; LEGACY_MODEL_ATTRIBUTE_ALIAS_COUNT]
{
    use crate::format::attrs as current_format;
    use crate::tokenizer::attrs as current_tokenizer;

    let alias = |label, historical, canonical| ModelAttributeAlias {
        label,
        historical,
        canonical,
    };

    [
        alias(
            "format.data",
            historical::data.id(),
            current_format::data.id(),
        ),
        alias(
            "format.data_f16",
            historical::data_f16.id(),
            current_format::data_f16.id(),
        ),
        alias(
            "format.shape",
            historical::shape.id(),
            current_format::shape.id(),
        ),
        alias(
            "format.data_q4",
            historical::data_q4.id(),
            current_format::data_q4.id(),
        ),
        alias(
            "format.data_q8",
            historical::data_q8.id(),
            current_format::data_q8.id(),
        ),
        alias(
            "format.q_scales",
            historical::q_scales.id(),
            current_format::q_scales.id(),
        ),
        alias(
            "format.format_marker",
            historical::format_marker.id(),
            current_format::format_marker.id(),
        ),
        alias(
            "format.weight",
            historical::weight.id(),
            current_format::weight.id(),
        ),
        alias(
            "format.bias",
            historical::bias.id(),
            current_format::bias.id(),
        ),
        alias(
            "format.kind",
            historical::kind.id(),
            current_format::kind.id(),
        ),
        alias(
            "format.safetensor_path",
            historical::safetensor_path.id(),
            current_format::safetensor_path.id(),
        ),
        alias(
            "format.index",
            historical::index.id(),
            current_format::index.id(),
        ),
        alias(
            "format.model_root",
            historical::model_root.id(),
            current_format::model_root.id(),
        ),
        alias(
            "format.member",
            historical::member.id(),
            current_format::member.id(),
        ),
        alias(
            "format.model_name",
            historical::model_name.id(),
            current_format::model_name.id(),
        ),
        alias(
            "format.source",
            historical::source.id(),
            current_format::source.id(),
        ),
        alias(
            "format.quantization",
            historical::quantization.id(),
            current_format::quantization.id(),
        ),
        alias(
            "gemma_lora.lora_rank",
            historical::lora_rank.id(),
            Attribute::<U256BE>::anchored(historical::lora_rank.id()).id(),
        ),
        alias(
            "gemma_lora.lora_alpha",
            historical::lora_alpha.id(),
            Attribute::<F64>::anchored(historical::lora_alpha.id()).id(),
        ),
        alias(
            "gemma_lora.lora_adapter",
            historical::lora_adapter.id(),
            Attribute::<inlineencodings::GenId>::anchored(historical::lora_adapter.id()).id(),
        ),
        alias(
            "gemma_lora.lora_projection",
            historical::lora_projection.id(),
            Attribute::<inlineencodings::ShortString>::anchored(historical::lora_projection.id())
                .id(),
        ),
        alias(
            "gemma_lora.lora_a",
            historical::lora_a.id(),
            Attribute::<inlineencodings::Handle<crate::format::F32Array>>::anchored(
                historical::lora_a.id(),
            )
            .id(),
        ),
        alias(
            "gemma_lora.lora_b",
            historical::lora_b.id(),
            Attribute::<inlineencodings::Handle<crate::format::F32Array>>::anchored(
                historical::lora_b.id(),
            )
            .id(),
        ),
        alias(
            "tokenizer.tokenizer",
            historical::tokenizer.id(),
            current_tokenizer::tokenizer.id(),
        ),
        alias(
            "tokenizer.vocab",
            historical::vocab.id(),
            current_tokenizer::vocab.id(),
        ),
        alias(
            "tokenizer.merge",
            historical::merge.id(),
            current_tokenizer::merge.id(),
        ),
        alias(
            "tokenizer.added",
            historical::added.id(),
            current_tokenizer::added.id(),
        ),
        alias(
            "tokenizer.normalizer",
            historical::normalizer.id(),
            current_tokenizer::normalizer.id(),
        ),
        alias(
            "tokenizer.pre_tokenizer",
            historical::pre_tokenizer.id(),
            current_tokenizer::pre_tokenizer.id(),
        ),
        alias(
            "tokenizer.post_processor",
            historical::post_processor.id(),
            current_tokenizer::post_processor.id(),
        ),
        alias(
            "tokenizer.decoder",
            historical::decoder.id(),
            current_tokenizer::decoder.id(),
        ),
        alias(
            "tokenizer.piece",
            historical::piece.id(),
            current_tokenizer::piece.id(),
        ),
        alias(
            "tokenizer.token_id",
            historical::token_id.id(),
            current_tokenizer::token_id.id(),
        ),
        alias(
            "tokenizer.piece_bytes",
            historical::piece_bytes.id(),
            current_tokenizer::piece_bytes.id(),
        ),
        alias(
            "tokenizer.merge_left",
            historical::merge_left.id(),
            current_tokenizer::merge_left.id(),
        ),
        alias(
            "tokenizer.merge_right",
            historical::merge_right.id(),
            current_tokenizer::merge_right.id(),
        ),
        alias(
            "tokenizer.unk_token",
            historical::unk_token.id(),
            current_tokenizer::unk_token.id(),
        ),
        alias(
            "tokenizer.continuing_subword_prefix",
            historical::continuing_subword_prefix.id(),
            current_tokenizer::continuing_subword_prefix.id(),
        ),
        alias(
            "tokenizer.end_of_word_suffix",
            historical::end_of_word_suffix.id(),
            current_tokenizer::end_of_word_suffix.id(),
        ),
        alias(
            "tokenizer.max_input_chars",
            historical::max_input_chars.id(),
            current_tokenizer::max_input_chars.id(),
        ),
        alias(
            "tokenizer.piece_score",
            historical::piece_score.id(),
            current_tokenizer::piece_score.id(),
        ),
        alias(
            "tokenizer.byte_fallback",
            historical::byte_fallback.id(),
            current_tokenizer::byte_fallback.id(),
        ),
        alias(
            "tokenizer.pattern",
            historical::pattern.id(),
            current_tokenizer::pattern.id(),
        ),
        alias(
            "tokenizer.replace_content",
            historical::replace_content.id(),
            current_tokenizer::replace_content.id(),
        ),
        alias(
            "tokenizer.behavior",
            historical::behavior.id(),
            current_tokenizer::behavior.id(),
        ),
    ]
}

/// Add canonical attribute aliases for every matching historical model fact.
///
/// The result is strictly additive: every input trible is retained, unknown
/// attributes are untouched, and an alias is inserted only when the exact
/// `(entity, canonical attribute, value)` trible is missing. Entity and value
/// bytes are copied unchanged. Running the projection over its own result is
/// therefore idempotent and reports zero additions.
pub fn project_legacy_model_attributes(facts: &TribleSet) -> ModelAttributeProjection {
    let aliases = legacy_model_attribute_aliases();
    let by_historical: HashMap<Id, usize> = aliases
        .iter()
        .enumerate()
        .map(|(index, alias)| (alias.historical, index))
        .collect();
    let mut mappings: Vec<_> = aliases
        .into_iter()
        .map(|alias| ModelAttributeAliasCounts {
            alias,
            historical_facts: 0,
            aliases_added: 0,
            aliases_already_present: 0,
        })
        .collect();
    let mut projected = facts.clone();
    let mut historical_facts = 0;
    let mut aliases_added = 0;

    for fact in facts {
        let Some(&mapping_index) = by_historical.get(fact.a()) else {
            continue;
        };
        let mapping = &mut mappings[mapping_index];
        mapping.historical_facts += 1;
        historical_facts += 1;

        let alias = Trible::force(
            fact.e(),
            &mapping.alias.canonical,
            fact.v::<UnknownInline>(),
        );
        if projected.contains(&alias) {
            mapping.aliases_already_present += 1;
        } else {
            projected.insert(&alias);
            mapping.aliases_added += 1;
            aliases_added += 1;
        }
    }

    ModelAttributeProjection {
        facts: projected,
        input_facts: facts.len(),
        historical_facts,
        aliases_added,
        mappings,
    }
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use triblespace::core::capability::capability_action;
    use triblespace::core::capability::policy::resource_policy;
    use triblespace::core::collection::records::{
        KIND_COLLECTION_DESCRIPTOR, collection_name, collection_representation,
    };
    use triblespace::core::collection::{read_capability, write_capability};
    use triblespace::core::repo::pile::Pile;

    use super::*;

    static NEXT_TEST_PILE: AtomicU64 = AtomicU64::new(0);

    struct TestPile(PathBuf);

    impl TestPile {
        fn new() -> Self {
            let ordinal = NEXT_TEST_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-collection-discovery-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create isolated collection discovery pile");
            Self(path)
        }
    }

    impl Drop for TestPile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn explicit_model_collection_uses_the_resident_frozen_store_view() {
        use triblespace::core::collection::CollectionStore;
        use triblespace::core::repo::memoryrepo::MemoryRepo;

        let mut repo = MemoryRepo::default();
        let key = SigningKey::from_bytes(&[0x70; 32]);
        let collection = repo
            .collection("models", direct_model_policy(key.verifying_key()))
            .unwrap();
        let member = fucid();
        let first = entity! { crate::format::attrs::member: &member };
        let expected = first.facts().clone();
        repo.commit(collection, &key, first).unwrap();

        // Records may arrive before their data. An unreadable later member
        // must not hide the models already available in this observation.
        let annotation = entity! { crate::format::attrs::parent: &member };
        let later: Blob<SimpleArchive> = annotation.into_facts().to_blob();
        let metadata = repo.put::<SimpleArchive, _>(TribleSet::new()).unwrap();
        repo.insert(CollectionRecord::Commit(CollectionCommit::sign(
            &key,
            collection.handle(),
            later.get_handle().transmute(),
            metadata,
        )))
        .unwrap();

        let before = repo.snapshot().unwrap();
        let attached = snapshot_model_collection_for(&before, collection).unwrap();
        assert_eq!(attached.facts(), &expected);
        assert_eq!(attached.support().len(), 1);
        assert_eq!(attached.support().collection(), collection);

        repo.put::<SimpleArchive, _>(later).unwrap();
        let after = repo.snapshot().unwrap();
        assert_eq!(
            snapshot_model_collection_for(&after, collection)
                .unwrap()
                .support()
                .len(),
            2
        );
        assert_eq!(
            snapshot_model_collection_for(&before, collection)
                .unwrap()
                .facts(),
            &expected
        );
    }

    #[test]
    fn named_collection_discovery_accepts_supported_plural_policies() {
        let path = TestPile::new();
        let mut pile = Pile::open(&path.0).unwrap();
        let first = SigningKey::from_bytes(&[0x71; 32]);
        let second = SigningKey::from_bytes(&[0x72; 32]);
        let descriptor = entity! {
            metadata::tag: KIND_COLLECTION_DESCRIPTOR,
            collection_name: mary_model_graph_name().to_owned(),
            collection_representation: SimpleArchive::id(),
            resource_policy*: AdmissionPolicy::direct(first.verifying_key()).binding(read_capability())
                + AdmissionPolicy::direct(second.verifying_key()).binding(read_capability()),
            resource_policy*: AdmissionPolicy::direct(first.verifying_key()).binding(write_capability())
                + AdmissionPolicy::direct(second.verifying_key()).binding(write_capability()),
        };
        // The scalar inspector is deliberately stricter than ordinary
        // admission. Discovery must not substitute it for supported rows.
        assert!(descriptor::policy(descriptor.facts()).is_err());
        for action in [ACTION_READ, ACTION_WRITE] {
            let definition = entity! { capability_action: action };
            pile.put::<SimpleArchive, _>(definition.facts().clone())
                .unwrap();
        }
        let policy_snapshot = pile.snapshot().unwrap();
        for action in [ACTION_READ, ACTION_WRITE] {
            assert_eq!(
                descriptor::admission_policies(
                    &policy_snapshot,
                    descriptor.facts(),
                    action,
                    Some(SimpleArchive::id()),
                )
                .count(),
                2,
            );
        }
        let collection = pile
            .register_collection::<SimpleArchive>(descriptor)
            .unwrap();
        let a = entity! { metadata::name: "first admitted leaf" };
        let b = entity! { metadata::name: "second admitted leaf" };
        let expected = a.facts().clone() + b.facts().clone();
        pile.commit(collection, &first, a).unwrap();
        pile.commit(collection, &second, b).unwrap();

        let snapshot = pile.snapshot().unwrap();
        let (current, retired) = named_collections_from_handles(
            &snapshot,
            mary_model_graph_name(),
            [collection.handle()],
        );
        assert_eq!(current, vec![collection]);
        assert!(retired.is_empty());
        assert_eq!(
            model_graph_collections_in(&snapshot).unwrap(),
            vec![collection],
        );
        let materialized = snapshot_model_collection_in(&snapshot).unwrap();
        assert_eq!(materialized.facts(), &expected);
        assert_eq!(materialized.support().len(), 2);
        drop(materialized);
        drop(snapshot);
        pile.close().unwrap();
    }

    #[test]
    fn named_collection_discovery_keeps_current_collection_without_admitted_commits() {
        let path = TestPile::new();
        let mut pile = Pile::open(&path.0).unwrap();
        let root = SigningKey::from_bytes(&[0x73; 32]);
        let outsider = SigningKey::from_bytes(&[0x74; 32]);
        let collection = pile
            .collection(
                mary_model_bundle_name(),
                direct_model_policy(root.verifying_key()),
            )
            .unwrap();

        // Even before any COMMIT references it, an explicitly supplied
        // current descriptor is not retired just because its support is empty.
        let snapshot = pile.snapshot().unwrap();
        let (current, retired) = named_collections_from_handles(
            &snapshot,
            mary_model_bundle_name(),
            [collection.handle()],
        );
        assert_eq!(current, vec![collection]);
        assert!(retired.is_empty());
        assert!(collection.admitted(&snapshot).unwrap().is_empty());
        drop(snapshot);

        // A valid signed claim supplies a discovery reference, not WRITE
        // authorization. This snapshot still admits no committed data.
        let claim = pile
            .commit(
                collection,
                &outsider,
                entity! { metadata::name: "unadmitted leaf" },
            )
            .unwrap();
        claim.verify_strict().unwrap();
        let snapshot = pile.snapshot().unwrap();
        assert!(
            !collection
                .writer_is_admitted(&snapshot, outsider.verifying_key())
                .unwrap()
        );
        assert_eq!(
            model_bundle_collections_in(&snapshot).unwrap(),
            vec![collection],
        );
        let materialized = snapshot_model_bundle_collection_in(&snapshot).unwrap();
        assert!(materialized.support().is_empty());
        assert!(materialized.facts().is_empty());
        drop(materialized);
        drop(snapshot);
        pile.close().unwrap();
    }
}
