//! Strictly additive migration from Mary's legacy branch-model piles.
//!
//! The legacy `main` branch remains authoritative, byte-for-byte evidence. A
//! migration captures one exact branch head, checks out only that head's
//! ancestry, projects the audited pre-epoch attribute aliases, adds the
//! requested canonical selection coordinates at the already-existing model
//! root, and publishes the resulting union as one native model-collection
//! commit. It never pushes, creates, deletes, or otherwise advances a branch.
//!
//! The policy-transfer bridge likewise preserves exact retired descriptors
//! and COMMIT records. It recognizes only the audited pre-authority,
//! mandatory-authority, and identical direct independent-policy Mary shapes;
//! it re-signs only valid claims authored by the descriptor's single root.
//!
//! Storage policy stays with the caller: this crate takes an already-open
//! [`Pile`], never reopens it, and neither flushes nor closes it. The caller
//! supplies the signing key and chooses the explicit durability boundary.
//!
//! # Why this is a crate and not a `mary` module
//!
//! The migration is a one-way bridge across the legacy branch -> Collection
//! cutover. `mary` itself is past that cutover, while TribleSpace retains only
//! the read-only pin snapshot and branch/commit encodings required to inspect
//! old piles. Keeping the bridge in a standalone package prevents those
//! migration-only concepts from becoming runtime dependencies again.

use std::collections::BTreeSet;

use anyhow::{anyhow, bail, Context};
use ed25519_dalek::{SigningKey, VerifyingKey};
use triblespace::core::blob::{Blob, IntoBlob, TryFromBlob};
use triblespace::core::collection::records::{
    collection_name, collection_representation, CollectionHandle, KIND_COLLECTION_DESCRIPTOR,
};
use triblespace::core::collection::{
    AdmissionPolicy, CollectionCommit, CollectionData, CollectionPolicy, CollectionRead,
    CollectionRecord, CollectionRecordSelector, CollectionStore, CollectionStoreExt, Cover,
};
use triblespace::core::inline::encodings::ed25519::ED25519PublicKey;
use triblespace::core::metadata::{self, MetaDescribe};
use triblespace::core::repo::memoryrepo::MemoryRepo;
use triblespace::core::repo::pile::{Pile, PileSnapshot};
use triblespace::core::repo::{
    self, content, parent, BlobStoreGet, CommitHandle, PinSnapshot, PinSnapshotSource,
    SnapshotSource,
};
use triblespace::prelude::blobencodings::UTF8String;
use triblespace::prelude::inlineencodings::{Handle, ShortString};
use triblespace::prelude::*;

use mary::format::attrs;
use mary::model_collection::{
    model_bundle_collection_or_create, model_graph_collection_or_create,
    prepare_model_bundle_fragment, project_legacy_model_attributes, publish_model_fragment,
    snapshot_model_bundle_collection_exact, ModelCollection,
};
use mary::models::personaplex::{PersonaPlexWeights, SOURCE as PERSONAPLEX_SOURCE};
use mary::selection::{select_model_root, select_tokenizer_root, ModelSelector, TokenizerSelector};

/// Every COMMIT naming one of `cover`'s members in its collection, whichever
/// key signed it: provenance, not admission. One `CommitMember` probe per
/// member on the record index, as the core's `Cover::commits` did until the
/// core dropped it (triblespace bfef707e); nothing is enumerated. Returned in
/// canonical record order, so no caller depends on how a store lays records
/// out.
fn member_commits<S>(
    cover: &Cover<blobencodings::SimpleArchive>,
    snapshot: &S,
) -> Result<Vec<CollectionCommit>, S::RecordsError>
where
    S: CollectionRead,
{
    let collection = cover.collection().handle();
    let selectors: BTreeSet<CollectionRecordSelector> = cover
        .members()
        .map(|member| CollectionRecordSelector::CommitMember(collection, Inline::new(member.raw)))
        .collect();
    if selectors.is_empty() {
        return Ok(Vec::new());
    }
    let mut commits: Vec<_> = snapshot
        .select_records(&selectors)?
        .into_iter()
        .filter_map(|record| match record {
            CollectionRecord::Commit(commit) => Some(commit),
            _ => None,
        })
        .collect();
    commits.sort();
    Ok(commits)
}

const PERSONAPLEX_LM_FILE: &str = "model.safetensors";
const PERSONAPLEX_MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const PERSONAPLEX_LM_MEMBERS: usize = 475;
const PERSONAPLEX_MIMI_MEMBERS: usize = 318;
const PERSONAPLEX_MEMBERS: usize = 793;

mod retired_collection {
    use super::*;

    attributes! {
        /// Mandatory authority of the descriptor epoch immediately preceding
        /// independent READ and WRITE policies.
        ///
        /// This is the original safe anchored declaration: its Ed25519
        /// encoding has always participated in the attribute identity.
        "7C31D328E9C369CCB6049D05CC8E8C77" as pub collection_authority:
            ED25519PublicKey;
    }
}

/// Published independent READ/WRITE policy links before capability-handle
/// bindings replaced them in Core cd8b2b7208b85bcb4de6169167a9bad07048e909.
/// These are the original safe anchors and GenId encodings from that commit's
/// parent, not new attributes or a runtime policy compatibility layer.
mod independent_policy_collection {
    use super::*;

    attributes! {
        "4108A59A03E8F8EC9DCDCC3C8597A292" as pub collection_read_policy:
            inlineencodings::GenId;
        "06930EAD5B83C83A30B6061B53A2840B" as pub collection_write_policy:
            inlineencodings::GenId;
    }
}

/// Exact schema of the named-root collection epoch before descriptors carried
/// an independent authority field.
///
/// These ids are migration evidence, not a runtime compatibility surface. The
/// declarations deliberately preserve the published byte identities so the
/// old descriptor can be reconstructed and compared by content address.
mod preauthority_collection {
    use super::*;

    pub const TRIBLE_SET_UNION_RECIPE_V1: Id =
        triblespace::macros::id_hex!("6D64C5F4B9E9B73F57C5F8702AB7FE45");
    pub const KIND_COLLECTION_RECIPE: Id =
        triblespace::macros::id_hex!("89E53D7FF204516307F0421C05E75000");

    attributes! {
        "436A04C372CBBFBD9C619CF50F59C4A1" unsafe as pub collection_name:
            ShortString;
        "6C1ED6495491E32FEBB9FDD4EE5E8907" unsafe as pub collection_namespace:
            ED25519PublicKey;
        "620FA4F2B456357DCD1882E583B85CC3" unsafe as pub collection_representation:
            inlineencodings::GenId;
        "5D338C58D897B969BE1AE0956CCFE301" unsafe as pub collection_recipe:
            inlineencodings::GenId;
    }
}

/// Which one of Mary's two root model collections an additive policy transfer
/// targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelCollectionKind {
    Graph,
    Bundle,
}

impl ModelCollectionKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Graph => mary::model_collection::mary_model_graph_name(),
            Self::Bundle => mary::model_collection::mary_model_bundle_name(),
        }
    }
}

/// Exact outcome of re-signing one retired collection's leaves under a current
/// direct-policy root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCollectionPolicyTransferResult {
    /// Explicit supported retired descriptor supplied by the caller.
    pub source: CollectionHandle,
    /// Current policy descriptor registered even when the source is empty.
    pub collection: ModelCollection,
    /// Authority recovered from, and covered by, the exact source descriptor.
    pub source_authority: VerifyingKey,
    /// Every deterministic successor COMMIT, including ones already present.
    pub commits: Vec<CollectionCommit>,
    /// Successor COMMITs newly appended by this invocation.
    pub appended_commits: usize,
    /// Old MERGE records deliberately left as rebuildable cache exhaust.
    pub skipped_merges: usize,
    /// Old DERIVE records deliberately left as rebuildable cache exhaust.
    pub skipped_derives: usize,
}

#[derive(Clone)]
struct PreparedPolicyTransfer {
    source: CollectionHandle,
    source_authority: VerifyingKey,
    collection: ModelCollection,
    commits: Vec<CollectionCommit>,
    missing: Vec<CollectionCommit>,
    skipped_merges: usize,
    skipped_derives: usize,
}

/// Data needed to turn one legacy model graph into a selectable native graph.
///
/// `model` resolves to exactly one entity carrying at least one `member` edge,
/// through the same exact-cardinality selectors the native runtime already
/// uses. [`ModelSelector::Name`] matches the canonical (possibly projected)
/// name and still fails closed when one pile holds two roots under one name;
/// [`ModelSelector::Root`] names the content address itself, which is how such
/// a pile states an unambiguous choice instead of being unmigratable. Neither
/// direction has a first-match fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegacyModelMigration<'a> {
    /// Which existing legacy weight root this migration publishes.
    pub model: ModelSelector<'a>,
    /// Canonical model-source coordinate to add or verify.
    pub source: &'a str,
    /// Canonical weight-format coordinate to add or verify.
    pub quantization: &'a str,
    /// If present, require exactly one tokenizer carrying this name.
    ///
    /// The audited legacy-attribute projection supplies the canonical
    /// `model_name` alias for pre-epoch tokenizers. This slice deliberately
    /// does not guess an unnamed tokenizer root.
    pub tokenizer_name: Option<&'a str>,
}

/// Exact evidence returned by one successful migration publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegacyModelMigrationResult {
    /// Signed native collection claim published by this migration.
    pub commit: CollectionCommit,
    /// Exact policy collection that received the commit.
    pub collection: ModelCollection,
    /// Legacy branch id whose head was frozen.
    pub legacy_branch: Id,
    /// Exact legacy commit head used for the checkout.
    pub legacy_head: CommitHandle,
    /// Existing model entity selected without recomputing its identity.
    pub model_root: Id,
    /// Existing tokenizer entity selected when `tokenizer_name` was supplied.
    pub tokenizer_root: Option<Id>,
    /// Fact count in the frozen legacy checkout.
    pub legacy_facts: usize,
    /// Missing canonical pre-epoch aliases added by the audited projection.
    pub aliases_added: usize,
    /// Missing explicit source/quantization coordinates added at the model root.
    pub selector_facts_added: usize,
}

/// One immutable read of the active legacy `main` branch.
///
/// The pin snapshot is consumed only to choose `branch` and `head`. `facts`
/// is then reconstructed from that exact head's ancestor closure through the
/// returned append-only blob reader, so a concurrent later branch advance
/// cannot widen the snapshot.
pub struct FrozenLegacyMain {
    /// Unique legacy branch carrying the name `main`.
    pub branch: Id,
    /// Exact signed branch head captured from the pin snapshot.
    pub head: CommitHandle,
    /// Union of every content archive reachable from `head`.
    pub facts: TribleSet,
    /// Immutable blob view resolving facts' attachment handles.
    pub reader: PileSnapshot,
}

/// Result of adopting the exact legacy PersonaPlex weight commit as one bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersonaPlexLegacyAdoptionResult {
    /// Existing bundle COMMIT on a no-op retry, or the newly published COMMIT.
    pub commit: CollectionCommit,
    /// Whether this invocation finalized a new COMMIT.
    pub published: bool,
    /// Exact policy collection that contains the token.
    pub collection: ModelCollection,
    /// Exact legacy commit-DAG node selected by the caller.
    pub legacy_commit: CommitHandle,
    /// Existing legacy LM model root selected by its exact file name.
    pub legacy_lm_root: Id,
    /// Existing legacy Mimi model root selected by its exact file name.
    pub legacy_mimi_root: Id,
    /// New intrinsic root derived only from the union of member edges.
    pub model_root: Id,
    /// Canonical model-fact archive `H` named by the one-row bundle token.
    pub model_archive_data: CollectionData,
    /// Fact count in the exact legacy checkout before alias projection.
    pub legacy_facts: usize,
    /// Missing canonical pre-epoch aliases added by the audited projection.
    pub aliases_added: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PersonaPlexMemberPolicy {
    legacy_facts: usize,
    lm: usize,
    mimi: usize,
    total: usize,
}

const PERSONAPLEX_MEMBER_POLICY: PersonaPlexMemberPolicy = PersonaPlexMemberPolicy {
    legacy_facts: 4_700,
    lm: PERSONAPLEX_LM_MEMBERS,
    mimi: PERSONAPLEX_MIMI_MEMBERS,
    total: PERSONAPLEX_MEMBERS,
};

fn direct_policy(root: VerifyingKey) -> CollectionPolicy {
    CollectionPolicy::new(AdmissionPolicy::direct(root), AdmissionPolicy::direct(root))
}

fn retired_collection_descriptor(name: &str, authority: VerifyingKey) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.to_owned(),
        retired_collection::collection_authority: authority,
        collection_representation*: <blobencodings::SimpleArchive as MetaDescribe>::describe(),
    }
}

fn independent_policy_collection_descriptor(name: &str, policy: &CollectionPolicy) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        collection_name: name.to_owned(),
        independent_policy_collection::collection_read_policy*: policy.read().fragment(),
        independent_policy_collection::collection_write_policy*: policy.write().fragment(),
        collection_representation*: <blobencodings::SimpleArchive as MetaDescribe>::describe(),
    }
}

fn preauthority_union_recipe_description() -> Fragment {
    let id = preauthority_collection::TRIBLE_SET_UNION_RECIPE_V1;
    entity! {
        ExclusiveId::force_ref(&id) @
            metadata::name: "trible-set-union-v1",
            metadata::description: "Set union of the tribles carried by a collection's elements. Associative, commutative and idempotent, so any two states have a least upper bound and merging is order-independent: a collection's value is the union over every element committed to it, and two replicas that have seen the same elements agree regardless of the order they arrived in. Takes no arguments.",
            metadata::tag: preauthority_collection::KIND_COLLECTION_RECIPE,
    }
}

fn preauthority_collection_descriptor(name: &str, namespace: VerifyingKey) -> Fragment {
    entity! {
        metadata::tag: KIND_COLLECTION_DESCRIPTOR,
        preauthority_collection::collection_name: name,
        preauthority_collection::collection_namespace: namespace,
        preauthority_collection::collection_representation*:
            <blobencodings::SimpleArchive as MetaDescribe>::describe(),
        preauthority_collection::collection_recipe*:
            preauthority_union_recipe_description(),
    }
}

fn collection_descriptor_handle(descriptor: &Fragment) -> CollectionHandle {
    let blob: Blob<blobencodings::SimpleArchive> = descriptor.facts().clone().to_blob();
    blob.get_handle()
}

fn decode_exact_retired_collection(
    snapshot: &PileSnapshot,
    source: CollectionHandle,
    kind: ModelCollectionKind,
) -> anyhow::Result<VerifyingKey> {
    let blob: Blob<blobencodings::SimpleArchive> = snapshot
        .get(source)
        .map_err(|error| anyhow!("read retired collection descriptor {source:?}: {error}"))?;
    let facts = TribleSet::try_from_blob(blob)
        .context("decode retired collection descriptor SimpleArchive")?;
    let descriptor = triblespace::core::collection::descriptor::entity(&facts)
        .context("find retired collection descriptor entity")?;
    if exists!((policy: Id), or!(
        pattern!(&facts, [{ descriptor @
            independent_policy_collection::collection_read_policy: ?policy,
        }]),
        pattern!(&facts, [{ descriptor @
            independent_policy_collection::collection_write_policy: ?policy,
        }]),
    )) {
        // This bridge recognizes only the historical Mary constructor:
        // identical direct, single-root READ and WRITE policies. A matching
        // root row is just a candidate; the complete descriptor hash below
        // also proves the thresholds, policy kinds, name, representation,
        // intrinsic identities, and absence of extra facts. In particular we
        // do not collapse independent roots, open policies, quorums, or
        // delegation evidence into an assumed root-only authorization law.
        let roots = find!(root: Inline<ED25519PublicKey>, pattern!(&facts, [
            { descriptor @
                independent_policy_collection::collection_read_policy: _?policy,
                independent_policy_collection::collection_write_policy: _?policy,
            },
            { _?policy @
                triblespace::core::capability::policy::admission_policy_root: ?root,
            },
        ]))
        .collect::<Vec<_>>();
        let [root] = roots.as_slice() else {
            bail!("source is not the exact direct independent-policy Mary descriptor");
        };
        let authority = VerifyingKey::from_bytes(&root.raw)
            .context("independent-policy collection root is invalid")?;
        // Use the fallible constructor: Ed25519 decoding alone does not
        // reject every noncanonical or weak capability principal.
        let admission = AdmissionPolicy::quorum([authority], 1, None)
            .context("independent-policy collection root is not a usable principal")?;
        let policy = CollectionPolicy::new(admission.clone(), admission);
        let expected = collection_descriptor_handle(&independent_policy_collection_descriptor(
            kind.name(),
            &policy,
        ));
        anyhow::ensure!(
            expected == source,
            "source is not the exact direct independent-policy Mary descriptor for {:?}",
            kind.name(),
        );
        return Ok(authority);
    }
    let authorities = facts
        .iter()
        .filter(|fact| {
            fact.e() == &descriptor && fact.a() == &retired_collection::collection_authority.id()
        })
        .map(|fact| *fact.v::<ED25519PublicKey>())
        .collect::<Vec<_>>();
    match authorities.as_slice() {
        [authority] => {
            let authority = VerifyingKey::from_bytes(&authority.raw)
                .map_err(|error| anyhow!("retired collection authority is invalid: {error}"))?;
            let name_handle = triblespace::core::collection::descriptor::name(&facts)
                .context("decode retired collection name")?
                .ok_or_else(|| anyhow!("retired collection descriptor has no name"))?;
            let name: anybytes::View<str> = snapshot
                .get::<anybytes::View<str>, UTF8String>(name_handle)
                .map_err(|error| anyhow!("read retired collection name: {error}"))?;
            anyhow::ensure!(
                &*name == kind.name(),
                "retired collection is named {:?}, expected {:?}",
                &*name,
                kind.name(),
            );

            // Reconstructing the complete predecessor descriptor is the
            // strict shape check. Any extra field or embedded entity changes
            // this content address and is rejected.
            let expected =
                collection_descriptor_handle(&retired_collection_descriptor(&name, authority));
            anyhow::ensure!(
                expected == source,
                "source is not the exact authority-era Mary descriptor for {:?}",
                kind.name(),
            );
            Ok(authority)
        }
        [] => {
            let namespaces = facts
                .iter()
                .filter(|fact| {
                    fact.e() == &descriptor
                        && fact.a() == &preauthority_collection::collection_namespace.id()
                })
                .map(|fact| *fact.v::<ED25519PublicKey>())
                .collect::<Vec<_>>();
            let namespace = match namespaces.as_slice() {
                [namespace] => VerifyingKey::from_bytes(&namespace.raw).map_err(|error| {
                    anyhow!("pre-authority collection namespace is invalid: {error}")
                })?,
                [] => bail!("retired collection descriptor has neither authority nor namespace"),
                _ => bail!("retired collection descriptor repeats collection_namespace"),
            };

            // In this one historical epoch the namespace key was both the
            // root-name discriminator and the only admitted writer. Rebuild
            // that complete descriptor—including its embedded schema and
            // union-law descriptions—before treating the namespace as the
            // transfer authority. This recognizes the exact old algebra; it
            // does not loosen migration to arbitrary named archives.
            let expected = collection_descriptor_handle(&preauthority_collection_descriptor(
                kind.name(),
                namespace,
            ));
            anyhow::ensure!(
                expected == source,
                "source is not the exact pre-authority Mary descriptor for {:?}",
                kind.name(),
            );
            Ok(namespace)
        }
        _ => bail!("retired collection descriptor repeats collection_authority"),
    }
}

fn current_collection_in_memory(
    kind: ModelCollectionKind,
    target_root: VerifyingKey,
) -> anyhow::Result<ModelCollection> {
    let mut scratch = MemoryRepo::default();
    scratch
        .collection(kind.name(), direct_policy(target_root))
        .map_err(|error| anyhow!("construct current {:?} descriptor: {error}", kind.name()))
}

fn prepare_policy_transfer(
    snapshot: &PileSnapshot,
    signing_key: &SigningKey,
    source: CollectionHandle,
    kind: ModelCollectionKind,
) -> anyhow::Result<PreparedPolicyTransfer> {
    let source_authority = decode_exact_retired_collection(snapshot, source, kind)?;
    let collection = current_collection_in_memory(kind, signing_key.verifying_key())?;
    let mut target_commits = BTreeSet::new();
    let mut commits = BTreeSet::new();
    let mut skipped_merges = 0usize;
    let mut skipped_derives = 0usize;

    for record in snapshot
        .records()
        .context("enumerate collection records for model policy transfer")?
    {
        let record = record.context("read collection record for model policy transfer")?;
        let record_collection = match record {
            CollectionRecord::Commit(commit) => commit.collection(),
            CollectionRecord::Merge(merge) => merge.collection(),
            CollectionRecord::Derive(derive) => derive.collection(),
        };
        if record_collection == collection.handle() {
            if let CollectionRecord::Commit(commit) = record {
                target_commits.insert(commit);
            }
        }
        if record_collection != source {
            continue;
        }
        match record {
            CollectionRecord::Commit(commit) => {
                let fingerprint = CollectionRecord::Commit(commit).fingerprint();
                commit.verify_strict().map_err(|error| {
                    anyhow!(
                        "retired model COMMIT fingerprint {fingerprint} has an invalid signature: {error}",
                    )
                })?;
                anyhow::ensure!(
                    commit.public_key().raw == source_authority.to_bytes(),
                    "retired model COMMIT fingerprint {fingerprint} was authored by a non-root writer; refusing to silently adopt delegated data",
                );
                let successor = CollectionCommit::sign(
                    signing_key,
                    collection.handle(),
                    commit.data(),
                    commit.metadata(),
                );
                commits.insert(successor);
            }
            CollectionRecord::Merge(_) => skipped_merges += 1,
            CollectionRecord::Derive(_) => skipped_derives += 1,
        }
    }

    let mut missing = Vec::new();
    for commit in &commits {
        if !target_commits.contains(commit) {
            missing.push(*commit);
        }
    }
    Ok(PreparedPolicyTransfer {
        source,
        source_authority,
        collection,
        commits: commits.into_iter().collect(),
        missing,
        skipped_merges,
        skipped_derives,
    })
}

/// Additively re-seat one exact supported retired Mary model collection under
/// the current direct READ/WRITE policy rooted at `signing_key`.
///
/// The old descriptor and records remain untouched. Each successor signs the
/// old COMMIT's exact data and metadata handles; no member blob is read or
/// copied. A source with no COMMITs still registers its current descriptor.
/// MERGE and DERIVE records are deliberately left as rebuildable cache
/// exhaust. Replay is idempotent because exact successor records form a set.
/// Independent-policy predecessors must have the same direct single-root
/// policy for both actions; other policy geometries are not silently adopted.
pub fn transfer_retired_model_collection(
    pile: &mut Pile,
    signing_key: &SigningKey,
    source: CollectionHandle,
    kind: ModelCollectionKind,
) -> anyhow::Result<ModelCollectionPolicyTransferResult> {
    pile.refresh()
        .context("refresh before model collection policy transfer")?;
    let snapshot = pile
        .snapshot()
        .context("freeze retired model collection transfer source")?;
    let prepared = prepare_policy_transfer(&snapshot, signing_key, source, kind)?;
    drop(snapshot);

    let collection = pile
        .collection(kind.name(), direct_policy(signing_key.verifying_key()))
        .map_err(|error| anyhow!("register current {:?} descriptor: {error}", kind.name()))?;
    anyhow::ensure!(
        collection == prepared.collection,
        "current model descriptor changed identity between preflight and publication",
    );
    let appended_commits = prepared.missing.len();
    for commit in &prepared.missing {
        pile.insert(CollectionRecord::Commit(*commit))
            .map_err(|error| anyhow!("append re-seated model COMMIT: {error}"))?;
    }

    // A fresh plan proves both that every deterministic successor is visible
    // and that no predecessor writer appended an untransferred suffix between
    // census and publication. Such a race is harmless: replay carries it.
    let snapshot = pile
        .snapshot()
        .context("freeze model policy transfer verification snapshot")?;
    let after = prepare_policy_transfer(&snapshot, signing_key, source, kind)?;
    anyhow::ensure!(
        after.missing.is_empty(),
        "model policy transfer observed {} new or missing successor COMMIT(s); replay after predecessor writers are quiescent",
        after.missing.len(),
    );
    let _: Blob<blobencodings::SimpleArchive> = snapshot
        .get(collection.handle())
        .context("read registered current model descriptor after transfer")?;

    Ok(ModelCollectionPolicyTransferResult {
        source: prepared.source,
        collection,
        source_authority: prepared.source_authority,
        commits: after.commits,
        appended_commits,
        skipped_merges: after.skipped_merges,
        skipped_derives: after.skipped_derives,
    })
}

fn freeze_legacy_main_from_snapshot(
    reader: PileSnapshot,
    pins: &PinSnapshot,
) -> anyhow::Result<FrozenLegacyMain> {
    let wanted_name = "main".to_owned().to_blob().get_handle();
    let mut matches = Vec::new();

    for raw in pins.iter_ordered() {
        let branch = Id::new(*raw).ok_or_else(|| anyhow!("legacy pin snapshot contains nil id"))?;
        let metadata_handle = *pins
            .get(raw)
            .expect("an ordered pin-snapshot key has one value");
        let metadata: TribleSet = reader
            .get(metadata_handle)
            .map_err(|error| anyhow!("read legacy branch {branch} metadata: {error}"))?;
        let Ok(subject) = repo::branch::branch_entity(&metadata, branch) else {
            // A legacy pin was a generic mechanism. Non-branch application
            // pins are not candidates for Mary's named `main` branch.
            continue;
        };

        let names: Vec<Inline<Handle<UTF8String>>> = metadata
            .iter()
            .filter(|fact| fact.e() == &subject && fact.a() == &metadata::name.id())
            .map(|fact| *fact.v::<Handle<UTF8String>>())
            .collect();
        match names.as_slice() {
            [name] if *name == wanted_name => matches.push((branch, subject, metadata)),
            names if names.contains(&wanted_name) => {
                bail!("legacy branch {branch} has ambiguous names including 'main'")
            }
            _ => {}
        }
    }

    let (branch, subject, metadata) = match matches.len() {
        0 => bail!("legacy model pile has no active 'main' branch"),
        1 => matches.pop().expect("one legacy main match"),
        count => bail!("legacy model pile has {count} active branches named 'main'"),
    };
    let heads: Vec<CommitHandle> = metadata
        .iter()
        .filter(|fact| fact.e() == &subject && fact.a() == &repo::head.id())
        .map(|fact| *fact.v::<Handle<blobencodings::SimpleArchive>>())
        .collect();
    let head = match heads.as_slice() {
        [] => bail!("legacy 'main' branch {branch} has no commits"),
        [head] => *head,
        _ => bail!("legacy 'main' branch {branch} has ambiguous heads"),
    };

    let head_blob: Blob<blobencodings::SimpleArchive> = reader
        .get(head)
        .map_err(|error| anyhow!("read legacy 'main' head {head:?}: {error}"))?;
    repo::branch::verify(branch, head_blob, metadata)
        .map_err(|_| anyhow!("legacy 'main' branch {branch} has an invalid head signature"))?;
    let facts = checkout_legacy_commit_ancestors(&reader, head)
        .with_context(|| format!("checkout frozen legacy 'main' head {head:?}"))?;

    Ok(FrozenLegacyMain {
        branch,
        head,
        facts,
        reader,
    })
}

/// Freeze the unique active legacy `main` branch without restoring a mutable
/// repository surface.
pub fn freeze_legacy_model_main(pile: &mut Pile) -> anyhow::Result<FrozenLegacyMain> {
    // Freeze names first, then open one later append-only blob view. Every
    // handle named by the point-in-time pin snapshot must predate that reader.
    let pins = pile
        .snapshot_pin_heads()
        .context("snapshot active legacy branch pins")?;
    let reader = pile
        .snapshot()
        .context("freeze legacy model pile snapshot")?;
    freeze_legacy_main_from_snapshot(reader, &pins)
}

/// Check out exactly `commit` and its ancestors from one immutable blob view.
///
/// This is the detached equivalent of the retired ancestor checkout:
/// every reachable commit contributes its optional content archive, while a
/// merge-only commit contributes no facts. It deliberately does not construct
/// a branch-oriented workspace, because pulling one would make an otherwise
/// resident exact commit depend on a live, well-formed branch pin.
fn checkout_legacy_commit_ancestors(
    reader: &impl BlobStoreGet,
    commit: CommitHandle,
) -> anyhow::Result<TribleSet> {
    let mut visited = BTreeSet::new();
    let mut pending = vec![commit];
    let mut facts = TribleSet::new();

    while let Some(commit) = pending.pop() {
        if !visited.insert(commit) {
            continue;
        }

        let metadata: TribleSet = reader
            .get(commit)
            .map_err(|error| anyhow!("read exact legacy commit {commit:?}: {error}"))?;
        let mut content_handles = find!(
            (content_handle: Inline<_>),
            pattern!(&metadata, [{ content: ?content_handle }])
        );
        let content_handle = content_handles.next().map(|(handle,)| handle);
        anyhow::ensure!(
            content_handles.next().is_none(),
            "exact legacy commit {commit:?} has ambiguous content"
        );
        if let Some(content_handle) = content_handle {
            let contribution: TribleSet = reader.get(content_handle).map_err(|error| {
                anyhow!("read content of exact legacy commit {commit:?}: {error}")
            })?;
            facts += contribution;
        }

        pending.extend(
            find!(
                (parent_handle: Inline<_>),
                pattern!(&metadata, [{ parent: ?parent_handle }])
            )
            .map(|(parent_handle,)| parent_handle),
        );
    }

    Ok(facts)
}

/// Check out exactly `commit` and its ancestors from the legacy pile.
///
/// The caller-supplied content handle is the complete checkout authority. No
/// branch id, branch metadata, or current head participates in the read.
fn freeze_legacy_commit(
    pile: &mut Pile,
    commit: CommitHandle,
) -> anyhow::Result<(TribleSet, PileSnapshot)> {
    pile.refresh()
        .context("refresh legacy model pile before exact commit checkout")?;

    let reader = pile
        .snapshot()
        .context("freeze exact legacy model snapshot")?;
    let facts = checkout_legacy_commit_ancestors(&reader, commit)
        .with_context(|| format!("checkout exact legacy commit {commit:?}"))?;
    Ok((facts, reader))
}

fn members_for_root(facts: &TribleSet, root: Id) -> BTreeSet<Id> {
    find!(
        (member: Id),
        pattern!(facts, [{ root @ attrs::member: ?member }])
    )
    .map(|(member,)| member)
    .collect()
}

struct PreparedPersonaPlexCandidate {
    fragment: Fragment,
    legacy_lm_root: Id,
    legacy_mimi_root: Id,
    model_root: Id,
    legacy_facts: usize,
    aliases_added: usize,
}

/// Construct the unpublished PersonaPlex bundle graph without reading or
/// copying tensor payloads.
fn prepare_legacy_personaplex_candidate(
    legacy: TribleSet,
    reader: &impl BlobStoreGet,
    policy: PersonaPlexMemberPolicy,
) -> anyhow::Result<PreparedPersonaPlexCandidate> {
    let legacy_facts = legacy.len();
    anyhow::ensure!(
        legacy_facts == policy.legacy_facts,
        "legacy PersonaPlex checkout has {legacy_facts} facts, expected exactly {}",
        policy.legacy_facts
    );
    let projection = project_legacy_model_attributes(&legacy);
    let aliases_added = projection.aliases_added;

    let legacy_lm_root = select_model_root(
        &projection.facts,
        reader,
        ModelSelector::Name(PERSONAPLEX_LM_FILE),
    )
    .context("select exactly one legacy PersonaPlex LM root")?;
    let legacy_mimi_root = select_model_root(
        &projection.facts,
        reader,
        ModelSelector::Name(PERSONAPLEX_MIMI_FILE),
    )
    .context("select exactly one legacy PersonaPlex Mimi root")?;
    anyhow::ensure!(
        legacy_lm_root != legacy_mimi_root,
        "PersonaPlex LM and Mimi names resolve to the same legacy root {legacy_lm_root}"
    );

    let lm_members = members_for_root(&projection.facts, legacy_lm_root);
    let mimi_members = members_for_root(&projection.facts, legacy_mimi_root);
    anyhow::ensure!(
        lm_members.len() == policy.lm,
        "legacy PersonaPlex LM root {legacy_lm_root} has {} members, expected {}",
        lm_members.len(),
        policy.lm
    );
    anyhow::ensure!(
        mimi_members.len() == policy.mimi,
        "legacy PersonaPlex Mimi root {legacy_mimi_root} has {} members, expected {}",
        mimi_members.len(),
        policy.mimi
    );
    if let Some(overlap) = lm_members.intersection(&mimi_members).next() {
        bail!("legacy PersonaPlex LM and Mimi roots share member {overlap}");
    }
    let members: BTreeSet<_> = lm_members.union(&mimi_members).copied().collect();
    anyhow::ensure!(
        members.len() == policy.total,
        "legacy PersonaPlex member union has {} members, expected {}",
        members.len(),
        policy.total
    );

    // Re-wrap raw projected facts with an intentionally empty attachment
    // store. Every historical tensor/name/shape handle remains a reference to
    // bytes already resident in the pile; only the new source label below
    // enters this Fragment's MemoryBlobStore.
    let mut fragment = Fragment::from_facts_and_blobs(
        projection.facts,
        triblespace::core::blob::MemoryBlobStore::new(),
    );
    let root = entity! { _ @ attrs::member*: members.iter() };
    let model_root = root.root().expect("non-empty member set yields one root");
    fragment += root;

    let source = fragment.put::<UTF8String, _>(PERSONAPLEX_SOURCE.to_owned());
    fragment += entity! { ExclusiveId::force_ref(&model_root) @
        attrs::source: source,
        attrs::quantization: mary::persist::QUANTIZATION_NATIVE,
    };
    anyhow::ensure!(
        fragment.root() == Some(model_root),
        "PersonaPlex candidate lost its unique intrinsic root export"
    );

    Ok(PreparedPersonaPlexCandidate {
        fragment,
        legacy_lm_root,
        legacy_mimi_root,
        model_root,
        legacy_facts,
        aliases_added,
    })
}

/// Adopt one exact legacy PersonaPlex commit-DAG node as a signed model bundle.
///
/// The commit handle supplied by the caller is the only legacy history
/// selector. The current `main` head cannot widen the checkout. Existing facts,
/// entity ids, and attachment handles are preserved byte-for-byte; the only new
/// graph identity is one root derived from the union of the two legacy member
/// sets. No `mary-model-graph` COMMIT is published.
///
/// The caller owns the already-open pile and its eventual close/flush boundary.
pub fn adopt_legacy_personaplex_bundle(
    pile: &mut Pile,
    signing_key: &SigningKey,
    legacy_commit: CommitHandle,
) -> anyhow::Result<PersonaPlexLegacyAdoptionResult> {
    adopt_legacy_personaplex_bundle_with_policy(
        pile,
        signing_key,
        legacy_commit,
        PERSONAPLEX_MEMBER_POLICY,
    )
}

fn adopt_legacy_personaplex_bundle_with_policy(
    pile: &mut Pile,
    signing_key: &SigningKey,
    legacy_commit: CommitHandle,
    policy: PersonaPlexMemberPolicy,
) -> anyhow::Result<PersonaPlexLegacyAdoptionResult> {
    let (legacy, reader) = freeze_legacy_commit(pile, legacy_commit)?;
    let candidate = prepare_legacy_personaplex_candidate(legacy, &reader, policy)?;
    drop(reader);

    let collection = model_bundle_collection_or_create(pile, signing_key)
        .context("select or create the PersonaPlex bundle collection")?;
    let prepared = prepare_model_bundle_fragment(candidate.model_root, candidate.fragment)
        .context("prepare canonical PersonaPlex bundle token")?;
    let model_archive_data = prepared.model_archive_data();

    // Freeze exactly the current collection's admitted support before staging any
    // dependency. A matching `(root, H)` makes the operation a strict no-op;
    // a different PersonaPlex identity fails before a COMMIT can be exposed.
    let observation = pile
        .snapshot()
        .context("freeze existing PersonaPlex bundle collection")?;
    let support = collection
        .admitted(&observation)
        .context("admit existing PersonaPlex bundle support")?;
    let snapshot = snapshot_model_bundle_collection_exact(&observation, &support)
        .context("materialize existing PersonaPlex bundle support")?;
    if let Some(existing) = PersonaPlexWeights::find_in_bundle_snapshot(snapshot)
        .context("inspect existing PersonaPlex bundle")?
    {
        anyhow::ensure!(
            existing.authority().model_root() == candidate.model_root
                && existing.authority().model_archive_data() == model_archive_data,
            "a different PersonaPlex bundle is already authoritative in this collection"
        );
        let token_data = existing.authority().bundle_token_data();
        // Support admits payloads, not every attestation of those payloads.
        // A retry must return a COMMIT whose writer is admitted by this same
        // frozen observation, even if another key also signed the token.
        let mut admitted_commit = None;
        for commit in member_commits(&support, &observation)
            .context("read existing PersonaPlex token provenance")?
        {
            if commit.data() != token_data {
                continue;
            }
            let writer = VerifyingKey::from_bytes(&commit.public_key().raw)
                .context("decode verified PersonaPlex token writer")?;
            if collection
                .writer_is_admitted(&observation, writer)
                .context("check existing PersonaPlex token writer admission")?
            {
                admitted_commit = Some(commit);
                break;
            }
        }
        let commit =
            admitted_commit.expect("validated bundle authority has an admitted commit root");
        return Ok(PersonaPlexLegacyAdoptionResult {
            commit,
            published: false,
            collection,
            legacy_commit,
            legacy_lm_root: candidate.legacy_lm_root,
            legacy_mimi_root: candidate.legacy_mimi_root,
            model_root: candidate.model_root,
            model_archive_data,
            legacy_facts: candidate.legacy_facts,
            aliases_added: candidate.aliases_added,
        });
    }
    drop(observation);

    let mut staged = prepared
        .into_prepared_commit()
        .stage_for(pile, collection, signing_key)
        .map_err(|error| anyhow!("stage PersonaPlex bundle dependencies: {error}"))?;

    // The source UTF8String is one of the just-staged dependencies, while the
    // legacy tensor blobs were already resident. Validate that exact combined
    // view through the same zero-copy resolver used by runtime loading before
    // the signed COMMIT becomes visible. Dropping `staged` on failure leaves
    // only inert content-addressed dependencies.
    let staged_reader = staged
        .store_mut()
        .snapshot()
        .context("freeze staged PersonaPlex dependencies")?;
    let archive: Blob<blobencodings::SimpleArchive> = staged_reader
        .get(inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(model_archive_data))
        .map_err(|error| anyhow!("read staged PersonaPlex model archive H: {error}"))?;
    let staged_facts =
        TribleSet::try_from_blob(archive).context("decode staged PersonaPlex model archive H")?;
    let weights = PersonaPlexWeights::from_graph(&staged_facts, staged_reader)
        .context("validate staged but unpublished legacy PersonaPlex candidate")?;
    anyhow::ensure!(
        weights.root() == candidate.model_root,
        "validated PersonaPlex root {} differs from constructed root {}",
        weights.root(),
        candidate.model_root
    );
    anyhow::ensure!(
        weights.count() == policy.total,
        "validated PersonaPlex candidate has {} unique tensors, expected {}",
        weights.count(),
        policy.total
    );
    drop(weights);

    let commit = staged
        .finalize()
        .map_err(|error| anyhow!("finalize validated PersonaPlex bundle: {error}"))?;

    Ok(PersonaPlexLegacyAdoptionResult {
        commit,
        published: true,
        collection,
        legacy_commit,
        legacy_lm_root: candidate.legacy_lm_root,
        legacy_mimi_root: candidate.legacy_mimi_root,
        model_root: candidate.model_root,
        model_archive_data,
        legacy_facts: candidate.legacy_facts,
        aliases_added: candidate.aliases_added,
    })
}

fn source_coordinate_is_missing(
    fragment: &Fragment,
    reader: &impl BlobStoreGet,
    root: Id,
    wanted: &str,
) -> anyhow::Result<bool> {
    let handles: Vec<Inline<Handle<UTF8String>>> = fragment
        .facts()
        .iter()
        .filter(|fact| fact.e() == &root && fact.a() == &attrs::source.id())
        .map(|fact| *fact.v::<Handle<UTF8String>>())
        .collect();

    match handles.as_slice() {
        [] => Ok(true),
        [handle] => {
            let actual: anybytes::View<str> = reader
                .get(*handle)
                .map_err(|error| anyhow!("read source on legacy model root {root}: {error}"))?;
            if &*actual != wanted {
                bail!(
                    "legacy model root {root} already has conflicting source {:?}",
                    &*actual
                );
            }
            Ok(false)
        }
        _ => bail!("legacy model root {root} has ambiguous source coordinates"),
    }
}

fn quantization_coordinate_is_missing(
    fragment: &Fragment,
    root: Id,
    wanted: &str,
) -> anyhow::Result<bool> {
    let values: Vec<Inline<ShortString>> = fragment
        .facts()
        .iter()
        .filter(|fact| fact.e() == &root && fact.a() == &attrs::quantization.id())
        .map(|fact| *fact.v::<ShortString>())
        .collect();

    match values.as_slice() {
        [] => {
            let _: Inline<ShortString> = wanted.try_to_inline().map_err(|error| {
                anyhow!("quantization {wanted:?} is not a valid ShortString coordinate: {error:?}")
            })?;
            Ok(true)
        }
        [value] => {
            let value = ShortString::validate(*value).map_err(|error| {
                anyhow!("invalid quantization bytes on legacy model root {root}: {error:?}")
            })?;
            let actual: &str = value.try_from_inline().map_err(|error| {
                anyhow!("invalid quantization UTF-8 on legacy model root {root}: {error}")
            })?;
            if actual != wanted {
                bail!("legacy model root {root} already has conflicting quantization {actual:?}");
            }
            Ok(false)
        }
        _ => bail!("legacy model root {root} has ambiguous quantization coordinates"),
    }
}

/// Publish one frozen legacy `main` snapshot into Mary's native collection.
///
/// The returned [`CollectionCommit`] is the exact signed claim just published.
/// Existing facts are copied as raw tribles, which preserves
/// entity ids, value bytes, unknown attributes, and resident attachment
/// handles. Only canonical aliases plus the requested selector coordinates are
/// added. The caller retains `pile` on both success and failure and must choose
/// its own explicit `flush`/`close` policy.
pub fn migrate_legacy_model_main(
    pile: &mut Pile,
    signing_key: &SigningKey,
    request: LegacyModelMigration<'_>,
) -> anyhow::Result<LegacyModelMigrationResult> {
    let pins = pile
        .snapshot_pin_heads()
        .context("snapshot active legacy branch pins")?;
    migrate_legacy_model_main_from_snapshot(pile, signing_key, &pins, request)
}

/// Internal seam for frozen legacy fixtures and the production pin census.
///
/// Supplying the whole immutable snapshot keeps tests honest without
/// reintroducing a mutable pin writer solely to manufacture history.
fn migrate_legacy_model_main_from_snapshot(
    pile: &mut Pile,
    signing_key: &SigningKey,
    pins: &PinSnapshot,
    request: LegacyModelMigration<'_>,
) -> anyhow::Result<LegacyModelMigrationResult> {
    let reader = pile
        .snapshot()
        .context("freeze legacy model pile snapshot")?;
    let frozen = freeze_legacy_main_from_snapshot(reader, pins)?;
    migrate_frozen_legacy_model(pile, signing_key, request, frozen)
}

fn migrate_frozen_legacy_model(
    pile: &mut Pile,
    signing_key: &SigningKey,
    request: LegacyModelMigration<'_>,
    frozen: FrozenLegacyMain,
) -> anyhow::Result<LegacyModelMigrationResult> {
    let legacy_facts = frozen.facts.len();
    let projection = project_legacy_model_attributes(&frozen.facts);
    let aliases_added = projection.aliases_added;

    let model_root = select_model_root(&projection.facts, &frozen.reader, request.model)
        .with_context(|| {
            format!(
                "select exactly one legacy weight root matching {:?}",
                request.model
            )
        })?;

    let tokenizer_root = request
        .tokenizer_name
        .map(|name| {
            select_tokenizer_root(
                &projection.facts,
                &frozen.reader,
                TokenizerSelector::Name(name),
            )
            .with_context(|| format!("select exactly one legacy tokenizer named {name:?}"))
        })
        .transpose()?;

    // Build at explicit, already-selected ids. No intrinsic entity id is ever
    // recomputed from the enriched coordinate set.
    let mut fragment = Fragment::from_facts_and_blobs(
        projection.facts,
        triblespace::core::blob::MemoryBlobStore::new(),
    );
    let add_source =
        source_coordinate_is_missing(&fragment, &frozen.reader, model_root, request.source)?;
    let add_quantization =
        quantization_coordinate_is_missing(&fragment, model_root, request.quantization)?;
    let selector_facts_added = usize::from(add_source) + usize::from(add_quantization);

    // Match Mary's ordinary model-root construction idiom: new coordinates
    // are described `entity!` facts at one forced existing id. The explicit
    // subject prevents intrinsic-id recomputation while the returned Fragment
    // carries the attributes' metafacts into the native commit metadata.
    if selector_facts_added != 0 {
        let source = add_source.then(|| fragment.put::<UTF8String, _>(request.source.to_owned()));
        let quantization = add_quantization.then_some(request.quantization);
        fragment += entity! { ExclusiveId::force_ref(&model_root) @
            attrs::source?: source,
            attrs::quantization?: quantization,
        };
    }

    let collection = model_graph_collection_or_create(pile, signing_key)
        .context("select or create the native model-graph policy collection")?;
    let commit = publish_model_fragment(pile, signing_key, fragment)
        .context("publish migrated model graph to Mary's native collection")?;

    Ok(LegacyModelMigrationResult {
        commit,
        collection,
        legacy_branch: frozen.branch,
        legacy_head: frozen.head,
        model_root,
        tokenizer_root,
        legacy_facts,
        aliases_added,
        selector_facts_added,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::OpenOptions;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use anybytes::{Bytes, View};
    use ed25519_dalek::Signer;
    use triblespace::core::blob::encodings::UnknownBlob;
    use triblespace::core::blob::MemoryBlobStore;
    use triblespace::core::inline::encodings::UnknownInline;
    use triblespace::core::metadata;
    use triblespace::core::patch::Entry;
    use triblespace::core::repo::pile::WantRewritePolicy;
    use triblespace::core::repo::{BlobStorePut, RetentionRoots};
    use triblespace::prelude::blobencodings::RawBytes;

    use super::*;
    use mary::format::{F32Array, U64Array};
    use mary::model_collection::{
        local_model_support, model_bundle_collections_in, model_graph_collections_in,
        publish_model_bundle_fragment, snapshot_model_bundle_collection_exact,
        snapshot_model_collection_for, snapshot_model_collection_named_in,
    };

    static NEXT_TEMP_PILE: AtomicU64 = AtomicU64::new(0);
    const LEGACY_MODEL_NAME: &str = "legacy-weights.safetensors";
    const CANONICAL_SOURCE: &str = "example/model-v1";
    const CANONICAL_TOKENIZER: &str = "example/model-v1";
    const QUANTIZATION: &str = "native";

    struct TempPilePath(PathBuf);

    impl TempPilePath {
        fn new(label: &str) -> Self {
            let ordinal = NEXT_TEMP_PILE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "mary-model-migration-{label}-{}-{nanos}-{ordinal}.pile",
                std::process::id()
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .expect("create disposable pile");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPilePath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    struct LegacyFixture {
        pile: TempPilePath,
        pins: PinSnapshot,
        branch: Id,
        head: CommitHandle,
        model_root: Id,
        /// The second root sharing `model_root`'s name, when the fixture was
        /// asked for the ambiguous shape.
        duplicate_root: Option<Id>,
        tokenizer_root: Id,
        attachment: Inline<Handle<UTF8String>>,
        unknown_fact: Trible,
        facts: TribleSet,
    }

    struct LegacyPersonaPlexFixture {
        pile: TempPilePath,
        weight_commit: CommitHandle,
        later_commit: CommitHandle,
        lm_root: Id,
        mimi_root: Id,
        facts: TribleSet,
        later_fact: Trible,
        leaf_ids: Vec<Id>,
        data: Vec<Inline<Handle<F32Array>>>,
        shapes: Vec<Inline<Handle<U64Array>>>,
        orphan: Inline<Handle<RawBytes>>,
    }

    fn key(byte: u8) -> SigningKey {
        SigningKey::from_bytes(&[byte; 32])
    }

    fn bytes32(hex: &str) -> [u8; 32] {
        assert_eq!(hex.len(), 64);
        let mut bytes = [0_u8; 32];
        for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
            let pair = std::str::from_utf8(pair).expect("test hex is ASCII");
            bytes[index] = u8::from_str_radix(pair, 16).expect("test hex is valid");
        }
        bytes
    }

    fn test_id(byte: u8) -> Id {
        Id::new([byte; 16]).expect("nonzero test id")
    }

    fn historical_attribute(label: &str) -> Id {
        mary::model_collection::legacy_model_attribute_aliases()
            .into_iter()
            .find(|alias| alias.label == label)
            .unwrap_or_else(|| panic!("missing legacy alias {label}"))
            .historical
    }

    fn replace_canonical_attributes_with_historical(facts: &TribleSet) -> TribleSet {
        let reverse: HashMap<Id, Id> = mary::model_collection::legacy_model_attribute_aliases()
            .into_iter()
            .map(|alias| (alias.canonical, alias.historical))
            .collect();
        facts
            .iter()
            .map(|fact| match reverse.get(fact.a()) {
                Some(historical) => Trible::force(fact.e(), historical, fact.v::<UnknownInline>()),
                None => *fact,
            })
            .collect()
    }

    /// Persist one ordinary legacy content/commit pair without manufacturing
    /// a mutable repository merely for the test fixture.
    fn write_legacy_commit(
        pile: &mut Pile,
        signer: &SigningKey,
        fragment: Fragment,
        parents: impl IntoIterator<Item = CommitHandle>,
    ) -> CommitHandle {
        let (facts, mut blobs) = fragment.into_facts_and_blobs();
        for (_, blob) in blobs.snapshot().expect("read fixture attachment store") {
            pile.put::<UnknownBlob, _>(blob)
                .expect("persist legacy fixture attachment");
        }

        let content_blob: Blob<blobencodings::SimpleArchive> = facts.to_blob();
        let content_handle = pile
            .put::<blobencodings::SimpleArchive, _>(content_blob.clone())
            .expect("persist legacy fixture content");
        let signature = signer.sign(&content_blob.bytes);
        let parents = parents.into_iter().collect::<Vec<_>>();
        let wrapper = entity! {
            repo::content: content_handle,
            repo::parent*: parents,
            triblespace::core::attestation::signed_by: signer.verifying_key(),
            triblespace::core::attestation::signature_r: signature,
            triblespace::core::attestation::signature_s: signature,
        };
        pile.put::<blobencodings::SimpleArchive, _>(wrapper.into_facts())
            .expect("persist legacy fixture commit")
    }

    fn write_branch_metadata(
        pile: &mut Pile,
        signer: &SigningKey,
        branch: Id,
        names: &[&str],
        heads: &[CommitHandle],
    ) -> Inline<Handle<blobencodings::SimpleArchive>> {
        let names = names
            .iter()
            .map(|name| {
                pile.put::<UTF8String, _>((*name).to_owned())
                    .expect("persist legacy branch name")
            })
            .collect::<Vec<_>>();
        let signed_head = *heads.first().expect("signed branch fixture has a head");
        let reader = pile.snapshot().expect("read branch fixture head");
        let head_blob: Blob<blobencodings::SimpleArchive> = reader
            .get(signed_head)
            .expect("resolve branch fixture head");
        let signature = signer.sign(&head_blob.bytes);
        let metadata = entity! {
            repo::branch: branch,
            repo::head*: heads.iter().copied(),
            metadata::name*: names,
            triblespace::core::attestation::signed_by: signer.verifying_key(),
            triblespace::core::attestation::signature_r: signature,
            triblespace::core::attestation::signature_s: signature,
        };
        pile.put::<blobencodings::SimpleArchive, _>(metadata.into_facts())
            .expect("persist legacy branch metadata")
    }

    fn insert_pin(
        pins: &mut PinSnapshot,
        branch: Id,
        metadata: Inline<Handle<blobencodings::SimpleArchive>>,
    ) {
        let raw: [u8; 16] = branch.into();
        pins.insert(&Entry::with_value(&raw, metadata));
    }

    fn legacy_fixture(label: &str, duplicate_model_name: bool) -> LegacyFixture {
        let pile = TempPilePath::new(label);
        let mut blobs = MemoryBlobStore::new();
        let tokenizer = mary::tokenizer::save_tokenizer_json(
            br###"{
                "model": {
                    "type": "WordPiece",
                    "vocab": {"[UNK]": 0, "hello": 1},
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100
                },
                "added_tokens": []
            }"###,
            CANONICAL_TOKENIZER,
            &mut blobs,
        )
        .expect("build legacy tokenizer fixture");
        let tokenizer_root = tokenizer.root().expect("tokenizer root");
        let mut facts = replace_canonical_attributes_with_historical(&tokenizer.into_facts());

        let model_root = test_id(0x11);
        let member = test_id(0x12);
        let name = blobs
            .put::<UTF8String, _>(LEGACY_MODEL_NAME.to_owned())
            .expect("put legacy model name");
        let attachment = blobs
            .put::<UTF8String, _>("resident legacy attachment".to_owned())
            .expect("put resident attachment");
        let member_value = inlineencodings::GenId::inline_from(member);
        let tokenizer_value = inlineencodings::GenId::inline_from(tokenizer_root);
        facts.insert(&Trible::force(
            &model_root,
            &historical_attribute("format.model_name"),
            &name,
        ));
        facts.insert(&Trible::force(
            &model_root,
            &historical_attribute("format.member"),
            &member_value,
        ));
        facts.insert(&Trible::force(
            &model_root,
            &historical_attribute("tokenizer.tokenizer"),
            &tokenizer_value,
        ));

        // This attribute is deliberately outside the audited mapping. Its
        // exact row and UTF8String handle must survive the native migration.
        let unknown_fact = Trible::force(&member, &test_id(0x7f), &attachment);
        facts.insert(&unknown_fact);

        let mut duplicate_root = None;
        if duplicate_model_name {
            let other_root = test_id(0x21);
            duplicate_root = Some(other_root);
            let other_member = inlineencodings::GenId::inline_from(test_id(0x22));
            facts.insert(&Trible::force(
                &other_root,
                &historical_attribute("format.model_name"),
                &name,
            ));
            facts.insert(&Trible::force(
                &other_root,
                &historical_attribute("format.member"),
                &other_member,
            ));
        }

        let fragment = Fragment::from_facts_and_blobs(facts.clone(), blobs);
        let mut pile_store = Pile::open(pile.path()).expect("open disposable pile");
        let legacy_signer = key(3);
        let head = write_legacy_commit(&mut pile_store, &legacy_signer, fragment, []);
        let branch = test_id(0x70);
        let branch_metadata =
            write_branch_metadata(&mut pile_store, &legacy_signer, branch, &["main"], &[head]);
        let mut pins = PinSnapshot::new();
        insert_pin(&mut pins, branch, branch_metadata);
        pile_store.close().expect("close legacy fixture pile");

        LegacyFixture {
            pile,
            pins,
            branch,
            head,
            model_root,
            duplicate_root,
            tokenizer_root,
            attachment,
            unknown_fact,
            facts,
        }
    }

    fn add_legacy_tensor(
        blobs: &mut MemoryBlobStore,
        facts: &mut TribleSet,
        root: Id,
        member: Id,
        leaf: Id,
        tensor_name: &str,
        value: f32,
    ) -> (Inline<Handle<F32Array>>, Inline<Handle<U64Array>>) {
        let data = blobs
            .put::<F32Array, _>(vec![value])
            .expect("put legacy f32 payload");
        let shape = blobs
            .put::<U64Array, _>(vec![1_u64])
            .expect("put legacy tensor shape");
        let path = blobs
            .put::<UTF8String, _>(tensor_name.to_owned())
            .expect("put legacy tensor path");

        facts.insert(&Trible::force(
            &leaf,
            &historical_attribute("format.data"),
            &data,
        ));
        facts.insert(&Trible::force(
            &leaf,
            &historical_attribute("format.shape"),
            &shape,
        ));
        facts.insert(&Trible::force(
            &member,
            &historical_attribute("format.safetensor_path"),
            &path,
        ));
        facts.insert(&Trible::force(
            &member,
            &historical_attribute("format.weight"),
            &inlineencodings::GenId::inline_from(leaf),
        ));
        facts.insert(&Trible::force(
            &root,
            &historical_attribute("format.member"),
            &inlineencodings::GenId::inline_from(member),
        ));
        (data, shape)
    }

    fn legacy_personaplex_fixture(
        label: &str,
        overlap: bool,
        duplicate_tensor_name: bool,
    ) -> LegacyPersonaPlexFixture {
        let pile = TempPilePath::new(label);
        let lm_root = test_id(0x31);
        let mimi_root = test_id(0x32);
        let mut facts = TribleSet::new();
        let mut blobs = MemoryBlobStore::new();
        let lm_name = blobs
            .put::<UTF8String, _>(PERSONAPLEX_LM_FILE.to_owned())
            .expect("put legacy LM name");
        let mimi_name = blobs
            .put::<UTF8String, _>(PERSONAPLEX_MIMI_FILE.to_owned())
            .expect("put legacy Mimi name");
        facts.insert(&Trible::force(
            &lm_root,
            &historical_attribute("format.model_name"),
            &lm_name,
        ));
        facts.insert(&Trible::force(
            &mimi_root,
            &historical_attribute("format.model_name"),
            &mimi_name,
        ));

        let lm_member_a = test_id(0x41);
        let lm_member_b = test_id(0x42);
        let lm_leaf_a = test_id(0x51);
        let lm_leaf_b = test_id(0x52);
        let mut leaf_ids = vec![lm_leaf_a, lm_leaf_b];
        let mut data = Vec::new();
        let mut shapes = Vec::new();
        let (payload, shape) = add_legacy_tensor(
            &mut blobs,
            &mut facts,
            lm_root,
            lm_member_a,
            lm_leaf_a,
            "lm.layer.0.weight",
            1.0,
        );
        data.push(payload);
        shapes.push(shape);
        let (payload, shape) = add_legacy_tensor(
            &mut blobs,
            &mut facts,
            lm_root,
            lm_member_b,
            lm_leaf_b,
            if duplicate_tensor_name {
                "lm.layer.0.weight"
            } else {
                "lm.layer.1.weight"
            },
            2.0,
        );
        data.push(payload);
        shapes.push(shape);

        if overlap {
            facts.insert(&Trible::force(
                &mimi_root,
                &historical_attribute("format.member"),
                &inlineencodings::GenId::inline_from(lm_member_a),
            ));
        } else {
            let mimi_member = test_id(0x43);
            let mimi_leaf = test_id(0x53);
            leaf_ids.push(mimi_leaf);
            let (payload, shape) = add_legacy_tensor(
                &mut blobs,
                &mut facts,
                mimi_root,
                mimi_member,
                mimi_leaf,
                "mimi.encoder.weight",
                3.0,
            );
            data.push(payload);
            shapes.push(shape);
        }

        let fragment = Fragment::from_facts_and_blobs(facts.clone(), blobs);
        let mut pile_store = Pile::open(pile.path()).expect("open PersonaPlex fixture pile");
        let legacy_signer = key(0x33);
        let weight_commit = write_legacy_commit(&mut pile_store, &legacy_signer, fragment, []);

        let mut later_blobs = MemoryBlobStore::new();
        let orphan = later_blobs
            .put::<RawBytes, _>(b"later unrelated orphan".to_vec())
            .expect("put later unrelated attachment");
        let later_fact = Trible::force(&test_id(0x61), &test_id(0x62), &orphan);
        let later_facts = std::iter::once(later_fact).collect();
        let later_commit = write_legacy_commit(
            &mut pile_store,
            &legacy_signer,
            Fragment::from_facts_and_blobs(later_facts, later_blobs),
            [weight_commit],
        );
        pile_store.close().expect("close PersonaPlex fixture pile");

        LegacyPersonaPlexFixture {
            pile,
            weight_commit,
            later_commit,
            lm_root,
            mimi_root,
            facts,
            later_fact,
            leaf_ids,
            data,
            shapes,
            orphan,
        }
    }

    fn personaplex_policy(fixture: &LegacyPersonaPlexFixture) -> PersonaPlexMemberPolicy {
        PersonaPlexMemberPolicy {
            legacy_facts: fixture.facts.len(),
            lm: 2,
            mimi: 1,
            total: 3,
        }
    }

    fn conflicting_personaplex_fragment() -> Fragment {
        let mut fragment = Fragment::empty();
        let shape = fragment.put::<U64Array, _>(vec![1_u64]);
        let data = fragment.put::<F32Array, _>(vec![99.0]);
        let leaf = entity! { _ @ attrs::data: data, attrs::shape: shape };
        let leaf_id = leaf.root().expect("conflicting tensor leaf");
        fragment += leaf;
        let name = fragment.put::<UTF8String, _>("other.weight".to_owned());
        let member = entity! { _ @ attrs::safetensor_path: name, attrs::weight: leaf_id };
        let member_id = member.root().expect("conflicting tensor member");
        fragment += member;
        let root = entity! { _ @ attrs::member: member_id };
        let root_id = root.root().expect("conflicting PersonaPlex root");
        fragment += root;
        let source = fragment.put::<UTF8String, _>(PERSONAPLEX_SOURCE.to_owned());
        fragment += entity! { ExclusiveId::force_ref(&root_id) @
            attrs::source: source,
            attrs::quantization: mary::persist::QUANTIZATION_NATIVE,
        };
        let (_, facts, metafacts, blobs) = fragment.into_parts();
        Fragment::rooted_from_parts(root_id, facts, metafacts, blobs)
    }

    fn read_main_identity(path: &Path, pins: &PinSnapshot) -> (Id, CommitHandle) {
        let mut pile = Pile::open(path).expect("open pile to inspect main");
        let reader = pile.snapshot().expect("snapshot inspected pile");
        let frozen = freeze_legacy_main_from_snapshot(reader, pins).expect("freeze legacy main");
        let identity = (frozen.branch, frozen.head);
        drop(frozen);
        pile.close().expect("close inspected pile");
        identity
    }

    fn exact_snapshot(
        path: &Path,
        collection: ModelCollection,
        commit: CollectionCommit,
    ) -> mary::model_collection::ModelPileSnapshot {
        let mut pile = Pile::open(path).expect("open pile for exact native read");
        let store = pile.snapshot().expect("freeze exact native read");
        // The typed descriptor is the exact value returned by publication; do
        // not replace it with a second ambient name-discovery pass.
        let snapshot = snapshot_model_collection_for(&store, collection)
            .expect("materialize admitted migration support");
        assert!(snapshot.support().contains(
            inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(commit.data())
        ));
        pile.close().expect("close exact-read pile");
        snapshot
    }

    fn exact_bundle_snapshot(
        path: &Path,
        collection: ModelCollection,
        commit: CollectionCommit,
    ) -> mary::model_collection::ModelPileSnapshot {
        let mut pile = Pile::open(path).expect("open pile for exact bundle read");
        let store = pile.snapshot().expect("freeze exact bundle read");
        let snapshot = snapshot_model_collection_for(&store, collection)
            .expect("materialize admitted model-bundle support");
        assert!(snapshot.support().contains(
            inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(commit.data())
        ));
        pile.close().expect("close exact bundle-read pile");
        snapshot
    }

    fn bundle_collections(pile: &mut Pile) -> Vec<ModelCollection> {
        let snapshot = pile.snapshot().expect("freeze bundle collection census");
        model_bundle_collections_in(&snapshot).expect("enumerate bundle collections")
    }

    fn put_descriptor(pile: &mut Pile, descriptor: &Fragment) -> CollectionHandle {
        let mut blobs = descriptor.blobs().clone();
        for (_, blob) in blobs.snapshot().unwrap() {
            pile.put::<UnknownBlob, _>(blob)
                .expect("persist descriptor dependency");
        }
        pile.put::<blobencodings::SimpleArchive, _>(descriptor.facts().clone())
            .expect("persist descriptor facts")
    }

    fn assert_policy_transfer_fails_before_publication(
        pile: &mut Pile,
        path: &Path,
        signing_key: &SigningKey,
        source: CollectionHandle,
        kind: ModelCollectionKind,
        expected_error: &str,
    ) {
        let target = current_collection_in_memory(kind, signing_key.verifying_key()).unwrap();
        let before = std::fs::metadata(path).unwrap().len();
        let error = transfer_retired_model_collection(pile, signing_key, source, kind)
            .expect_err("malformed retired collection must fail");
        assert!(
            format!("{error:#}").contains(expected_error),
            "unexpected migration error: {error:#}"
        );
        assert_eq!(
            std::fs::metadata(path).unwrap().len(),
            before,
            "failed transfer appended to the pile"
        );
        let snapshot = pile.snapshot().unwrap();
        let descriptor: Result<Blob<blobencodings::SimpleArchive>, _> =
            snapshot.get(target.handle());
        assert!(
            descriptor.is_err(),
            "failed transfer published its target descriptor"
        );
    }

    #[test]
    fn preauthority_descriptor_reconstructs_the_frozen_inkling_identity() {
        let namespace = VerifyingKey::from_bytes(&bytes32(
            "622f1356348389185f3bc07e04ad0de50bdb4344194522b850271d5b42a22609",
        ))
        .unwrap();
        let descriptor =
            preauthority_collection_descriptor(ModelCollectionKind::Graph.name(), namespace);
        assert_eq!(descriptor.facts().len(), 11);
        assert_eq!(
            collection_descriptor_handle(&descriptor).raw,
            bytes32("dd89e395ddc466e3ff5e9d002e4e7feef4b0055fb55c6cbdd9e9d3a3e96b0417"),
        );
    }

    #[test]
    fn independent_policy_descriptor_reconstructs_the_frozen_inkling_identity() {
        // Descriptor census of inkling-small-complete-policy-20260831.pile:
        // 704 bytes / 11 facts, two old action links to one direct policy.
        // This test never opens the production checkpoint.
        let authority = VerifyingKey::from_bytes(&bytes32(
            "622f1356348389185f3bc07e04ad0de50bdb4344194522b850271d5b42a22609",
        ))
        .unwrap();
        let descriptor = independent_policy_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            &direct_policy(authority),
        );
        assert_eq!(descriptor.facts().len(), 11);
        assert_eq!(
            collection_descriptor_handle(&descriptor).raw,
            bytes32("47f12177261336552ea95058671a5c77b27ee28056ed054a177d451ee7ba83a3"),
        );
    }

    #[test]
    fn independent_policy_transfer_preserves_evidence_admits_leaves_and_replays_exactly() {
        for kind in [ModelCollectionKind::Graph, ModelCollectionKind::Bundle] {
            let path = TempPilePath::new("independent-policy-transfer");
            let old = key(0x61);
            let new = key(0x62);
            let descriptor = independent_policy_collection_descriptor(
                kind.name(),
                &direct_policy(old.verifying_key()),
            );
            let mut pile = Pile::open(path.path()).unwrap();
            let source = put_descriptor(&mut pile, &descriptor);
            let mut retired = Vec::new();
            let mut expected_facts = TribleSet::new();
            for marker in [0x63, 0x64] {
                let facts = entity! { metadata::tag: test_id(marker) }.into_facts();
                let metadata = pile
                    .put::<blobencodings::SimpleArchive, _>(entity! {
                        metadata::name: format!("source evidence {marker}"),
                    })
                    .unwrap();
                let data_handle = pile
                    .put::<blobencodings::SimpleArchive, _>(facts.clone())
                    .unwrap();
                expected_facts += facts;
                let data =
                    inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(data_handle);
                let commit = CollectionCommit::sign(&old, source, data, metadata);
                pile.insert(CollectionRecord::Commit(commit)).unwrap();
                retired.push(commit);
            }
            let before = std::fs::read(path.path()).unwrap();
            let snapshot = pile.snapshot().unwrap();
            let historical = ModelCollection::open(&snapshot, source).unwrap();
            assert!(historical.admitted(&snapshot).unwrap().is_empty());
            let discovery = match kind {
                ModelCollectionKind::Graph => model_graph_collections_in(&snapshot),
                ModelCollectionKind::Bundle => model_bundle_collections_in(&snapshot),
            };
            assert!(discovery
                .unwrap_err()
                .to_string()
                .contains("found only retired descriptors"));
            drop(snapshot);

            let first = transfer_retired_model_collection(&mut pile, &new, source, kind).unwrap();
            assert_eq!(first.source_authority, old.verifying_key());
            assert_ne!(first.collection.handle(), source);
            assert_eq!(first.appended_commits, retired.len());
            let successors: BTreeSet<_> = retired
                .iter()
                .map(|commit| {
                    CollectionCommit::sign(
                        &new,
                        first.collection.handle(),
                        commit.data(),
                        commit.metadata(),
                    )
                })
                .collect();
            assert_eq!(
                first.commits.iter().copied().collect::<BTreeSet<_>>(),
                successors,
            );
            assert!(std::fs::read(path.path()).unwrap().starts_with(&before));

            let snapshot = pile.snapshot().unwrap();
            let source_blob: Blob<blobencodings::SimpleArchive> = snapshot.get(source).unwrap();
            assert_eq!(
                &TribleSet::try_from_blob(source_blob).unwrap(),
                descriptor.facts(),
            );
            let records: Vec<_> = snapshot.records().unwrap().map(Result::unwrap).collect();
            for commit in &retired {
                assert!(records.contains(&CollectionRecord::Commit(*commit)));
            }
            let discovered = match kind {
                ModelCollectionKind::Graph => model_graph_collections_in(&snapshot),
                ModelCollectionKind::Bundle => model_bundle_collections_in(&snapshot),
            }
            .unwrap();
            assert_eq!(discovered, vec![first.collection]);
            // Ordinary admission remains unchanged: old links admit nothing,
            // while the explicit successor descriptor admits precisely the
            // re-signed data leaves under its new capability policy.
            assert!(historical.admitted(&snapshot).unwrap().is_empty());
            let materialized = snapshot_model_collection_named_in(&snapshot, kind.name()).unwrap();
            assert_eq!(materialized.facts(), &expected_facts);
            assert_eq!(materialized.support().len(), retired.len());
            drop(materialized);
            drop(snapshot);

            let after = std::fs::read(path.path()).unwrap();
            let replay = transfer_retired_model_collection(&mut pile, &new, source, kind).unwrap();
            assert_eq!(replay.commits, first.commits);
            assert_eq!(replay.appended_commits, 0);
            assert_eq!(std::fs::read(path.path()).unwrap(), after);
            pile.close().unwrap();
        }
    }

    #[test]
    fn independent_policy_transfer_rejects_other_policy_geometries_before_publication() {
        let old = key(0x65).verifying_key();
        let other = key(0x66).verifying_key();
        let policies = [
            CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::Open),
            CollectionPolicy::new(AdmissionPolicy::Open, AdmissionPolicy::direct(old)),
            CollectionPolicy::new(AdmissionPolicy::direct(old), AdmissionPolicy::direct(other)),
            CollectionPolicy::new(
                AdmissionPolicy::delegable(old),
                AdmissionPolicy::delegable(old),
            ),
            CollectionPolicy::new(
                AdmissionPolicy::quorum([old, other], 2, None).unwrap(),
                AdmissionPolicy::quorum([old, other], 2, None).unwrap(),
            ),
        ];
        for policy in policies {
            let path = TempPilePath::new("independent-policy-other-geometry");
            let mut pile = Pile::open(path.path()).unwrap();
            let descriptor =
                independent_policy_collection_descriptor(ModelCollectionKind::Graph.name(), &policy);
            let source = put_descriptor(&mut pile, &descriptor);
            assert_policy_transfer_fails_before_publication(
                &mut pile,
                path.path(),
                &key(0x67),
                source,
                ModelCollectionKind::Graph,
                "source is not the exact direct independent-policy Mary descriptor",
            );
            pile.close().unwrap();
        }
    }

    #[test]
    fn independent_policy_transfer_rejects_extensions_and_wrong_kind_before_publication() {
        let base = independent_policy_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            &direct_policy(key(0x68).verifying_key()),
        );
        let mut extended = base.clone();
        let root = base.root().unwrap();
        extended += entity! {
            ExclusiveId::force_ref(&root) @ metadata::description: "unexpected extension",
        };
        let mut extra_entity = base.clone();
        extra_entity += entity! { metadata::name: "unlinked policy evidence" };
        for (descriptor, kind) in [
            (extended, ModelCollectionKind::Graph),
            (extra_entity, ModelCollectionKind::Graph),
            (base, ModelCollectionKind::Bundle),
        ] {
            let path = TempPilePath::new("independent-policy-wrong-shape");
            let mut pile = Pile::open(path.path()).unwrap();
            let source = put_descriptor(&mut pile, &descriptor);
            assert_policy_transfer_fails_before_publication(
                &mut pile,
                path.path(),
                &key(0x69),
                source,
                kind,
                "source is not the exact direct independent-policy Mary descriptor",
            );
            pile.close().unwrap();
        }
    }

    #[test]
    fn independent_policy_transfer_rejects_a_weak_root_without_panicking() {
        use triblespace::core::capability::policy::{
            admission_invoke_threshold, admission_policy_root, KIND_ADMISSION_POLICY_QUORUM,
        };

        let mut identity = [0_u8; 32];
        identity[0] = 1;
        let weak = VerifyingKey::from_bytes(&identity).unwrap();
        let policy = entity! {
            metadata::tag: KIND_ADMISSION_POLICY_QUORUM,
            admission_policy_root: weak,
            admission_invoke_threshold: 1_u32,
        };
        let descriptor = entity! {
            metadata::tag: KIND_COLLECTION_DESCRIPTOR,
            collection_name: ModelCollectionKind::Graph.name().to_owned(),
            independent_policy_collection::collection_read_policy*: policy.clone(),
            independent_policy_collection::collection_write_policy*: policy,
            collection_representation*: <blobencodings::SimpleArchive as MetaDescribe>::describe(),
        };
        let path = TempPilePath::new("independent-policy-weak-root");
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);
        assert_policy_transfer_fails_before_publication(
            &mut pile,
            path.path(),
            &key(0x6E),
            source,
            ModelCollectionKind::Graph,
            "independent-policy collection root is not a usable principal",
        );
        pile.close().unwrap();
    }

    #[test]
    fn independent_policy_transfer_rejects_invalid_and_non_root_commits_before_publication() {
        for invalid_signature in [false, true] {
            let path = TempPilePath::new("independent-policy-invalid-source");
            let old = key(0x6A);
            let other = key(0x6B);
            let new = key(0x6C);
            let descriptor = independent_policy_collection_descriptor(
                ModelCollectionKind::Graph.name(),
                &direct_policy(old.verifying_key()),
            );
            let mut pile = Pile::open(path.path()).unwrap();
            let source = put_descriptor(&mut pile, &descriptor);
            let metadata = pile
                .put::<blobencodings::SimpleArchive, _>(TribleSet::new())
                .unwrap();
            let data_handle = pile
                .put::<blobencodings::SimpleArchive, _>(entity! { metadata::tag: test_id(0x6D) })
                .unwrap();
            let data = inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(data_handle);
            let valid = CollectionCommit::sign(&old, source, data, metadata);
            pile.insert(CollectionRecord::Commit(valid)).unwrap();
            if invalid_signature {
                // The checked foreign decoder rejects bad signatures, so this
                // fixture cannot mint an unverifiable COMMIT in memory any
                // more. Corrupt the final persisted COMMIT instead, so the
                // migration audits local evidence that arrived through the
                // trusted persisted-ingress decoder — the same shape as the
                // pre-authority sibling fixture above.
                let mut bytes = valid.to_bytes();
                bytes[5 * 32] ^= 1;
                assert!(CollectionCommit::from_bytes(bytes).is_err());
                pile.close().unwrap();
                let mut raw = std::fs::read(path.path()).unwrap();
                *raw.last_mut().unwrap() ^= 1;
                std::fs::write(path.path(), raw).unwrap();
                pile = Pile::open(path.path()).unwrap();
            } else {
                let commit = CollectionCommit::sign(&other, source, data, metadata);
                commit.verify_strict().unwrap();
                pile.insert(CollectionRecord::Commit(commit)).unwrap();
            }
            assert_policy_transfer_fails_before_publication(
                &mut pile,
                path.path(),
                &new,
                source,
                ModelCollectionKind::Graph,
                if invalid_signature {
                    "invalid signature"
                } else {
                    "authored by a non-root writer"
                },
            );
            pile.close().unwrap();
        }
    }

    #[test]
    fn policy_transfer_accepts_the_exact_preauthority_epoch() {
        let path = TempPilePath::new("preauthority-policy-transfer");
        let old = key(0x40);
        let new = key(0x41);
        let descriptor = preauthority_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            old.verifying_key(),
        );
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);

        let metadata = pile
            .put::<blobencodings::SimpleArchive, _>(TribleSet::new())
            .unwrap();
        let data_handle = pile
            .put::<blobencodings::SimpleArchive, _>(entity! { metadata::tag: test_id(0x42) })
            .unwrap();
        let data = inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(data_handle);
        let retired = CollectionCommit::sign(&old, source, data, metadata);
        pile.insert(CollectionRecord::Commit(retired)).unwrap();

        let result =
            transfer_retired_model_collection(&mut pile, &new, source, ModelCollectionKind::Graph)
                .unwrap();
        assert_eq!(result.source_authority, old.verifying_key());
        assert_eq!(result.appended_commits, 1);
        assert_eq!(result.commits[0].data(), retired.data());
        assert_eq!(result.commits[0].metadata(), retired.metadata());
        pile.close().unwrap();
    }

    #[test]
    fn policy_transfer_rejects_an_extended_preauthority_descriptor_before_publication() {
        let path = TempPilePath::new("preauthority-extended-descriptor");
        let old = key(0x50);
        let new = key(0x51);
        let mut descriptor = preauthority_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            old.verifying_key(),
        );
        let root = descriptor.root().unwrap();
        descriptor += entity! {
            ExclusiveId::force_ref(&root) @ metadata::description: "unexpected extension",
        };
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);

        assert_policy_transfer_fails_before_publication(
            &mut pile,
            path.path(),
            &new,
            source,
            ModelCollectionKind::Graph,
            "source is not the exact pre-authority Mary descriptor",
        );
        pile.close().unwrap();
    }

    #[test]
    fn policy_transfer_rejects_the_wrong_model_kind_before_publication() {
        let path = TempPilePath::new("preauthority-wrong-kind");
        let old = key(0x52);
        let new = key(0x53);
        let descriptor = preauthority_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            old.verifying_key(),
        );
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);

        assert_policy_transfer_fails_before_publication(
            &mut pile,
            path.path(),
            &new,
            source,
            ModelCollectionKind::Bundle,
            "source is not the exact pre-authority Mary descriptor",
        );
        pile.close().unwrap();
    }

    #[test]
    fn policy_transfer_rejects_an_invalid_source_signature_before_publication() {
        let path = TempPilePath::new("preauthority-invalid-signature");
        let old = key(0x54);
        let new = key(0x55);
        let descriptor = preauthority_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            old.verifying_key(),
        );
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);
        let metadata = pile
            .put::<blobencodings::SimpleArchive, _>(TribleSet::new())
            .unwrap();
        let data_handle = pile
            .put::<blobencodings::SimpleArchive, _>(entity! { metadata::tag: test_id(0x56) })
            .unwrap();
        let data = inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(data_handle);
        let valid = CollectionCommit::sign(&old, source, data, metadata);
        let mut bytes = valid.to_bytes();
        bytes[5 * 32] ^= 1;
        assert!(CollectionCommit::from_bytes(bytes).is_err());
        pile.insert(CollectionRecord::Commit(valid)).unwrap();
        pile.close().unwrap();

        // The checked foreign decoder rejects bad signatures. Corrupt this
        // fixture's final persisted COMMIT so the migration audits local evidence.
        let mut raw = std::fs::read(path.path()).unwrap();
        *raw.last_mut().unwrap() ^= 1;
        std::fs::write(path.path(), raw).unwrap();
        let mut pile = Pile::open(path.path()).unwrap();

        assert_policy_transfer_fails_before_publication(
            &mut pile,
            path.path(),
            &new,
            source,
            ModelCollectionKind::Graph,
            "invalid signature",
        );
        pile.close().unwrap();
    }

    #[test]
    fn policy_transfer_rejects_a_valid_non_root_writer_before_publication() {
        let path = TempPilePath::new("preauthority-non-root-writer");
        let old = key(0x57);
        let delegated = key(0x58);
        let new = key(0x59);
        let descriptor = preauthority_collection_descriptor(
            ModelCollectionKind::Graph.name(),
            old.verifying_key(),
        );
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);
        let metadata = pile
            .put::<blobencodings::SimpleArchive, _>(TribleSet::new())
            .unwrap();
        let data_handle = pile
            .put::<blobencodings::SimpleArchive, _>(entity! { metadata::tag: test_id(0x5A) })
            .unwrap();
        let data = inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(data_handle);
        let commit = CollectionCommit::sign(&delegated, source, data, metadata);
        commit.verify_strict().unwrap();
        pile.insert(CollectionRecord::Commit(commit)).unwrap();

        assert_policy_transfer_fails_before_publication(
            &mut pile,
            path.path(),
            &new,
            source,
            ModelCollectionKind::Graph,
            "authored by a non-root writer",
        );
        pile.close().unwrap();
    }

    #[test]
    fn policy_transfer_preserves_handles_registers_descriptor_and_replays_exactly() {
        let path = TempPilePath::new("policy-transfer");
        let old = key(0x41);
        let new = key(0x42);
        let descriptor =
            retired_collection_descriptor(ModelCollectionKind::Graph.name(), old.verifying_key());
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);
        assert_eq!(source, collection_descriptor_handle(&descriptor));

        let metadata = pile
            .put::<blobencodings::SimpleArchive, _>(TribleSet::new())
            .unwrap();
        let data_handle = pile
            .put::<blobencodings::SimpleArchive, _>(entity! { metadata::tag: test_id(0x43) })
            .unwrap();
        let data = inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(data_handle);
        let retired = CollectionCommit::sign(&old, source, data, metadata);
        pile.insert(CollectionRecord::Commit(retired)).unwrap();

        let first =
            transfer_retired_model_collection(&mut pile, &new, source, ModelCollectionKind::Graph)
                .unwrap();
        assert_eq!(first.source_authority, old.verifying_key());
        assert_eq!(first.appended_commits, 1);
        assert_eq!(first.commits.len(), 1);
        assert_eq!(first.commits[0].data(), retired.data());
        assert_eq!(first.commits[0].metadata(), retired.metadata());
        assert_eq!(
            first.commits[0].public_key().raw,
            new.verifying_key().to_bytes()
        );

        let snapshot = pile.snapshot().unwrap();
        assert_eq!(
            ModelCollection::open(&snapshot, first.collection.handle()).unwrap(),
            first.collection,
        );
        let support = first.collection.admitted(&snapshot).unwrap();
        let admitted: Vec<_> = member_commits(&support, &snapshot)
            .unwrap()
            .into_iter()
            .filter(|commit| {
                let writer = VerifyingKey::from_bytes(&commit.public_key().raw).unwrap();
                first
                    .collection
                    .writer_is_admitted(&snapshot, writer)
                    .unwrap()
            })
            .collect();
        assert_eq!(admitted, first.commits);
        drop(snapshot);

        let len = std::fs::metadata(path.path()).unwrap().len();
        let replay =
            transfer_retired_model_collection(&mut pile, &new, source, ModelCollectionKind::Graph)
                .unwrap();
        assert_eq!(replay.commits, first.commits);
        assert_eq!(replay.appended_commits, 0);
        assert_eq!(std::fs::metadata(path.path()).unwrap().len(), len);
        pile.close().unwrap();
    }

    #[test]
    fn policy_transfer_registers_an_empty_current_collection() {
        let path = TempPilePath::new("empty-policy-transfer");
        let old = key(0x44);
        let new = key(0x45);
        let descriptor =
            retired_collection_descriptor(ModelCollectionKind::Bundle.name(), old.verifying_key());
        let mut pile = Pile::open(path.path()).unwrap();
        let source = put_descriptor(&mut pile, &descriptor);

        let result =
            transfer_retired_model_collection(&mut pile, &new, source, ModelCollectionKind::Bundle)
                .unwrap();
        assert!(result.commits.is_empty());
        assert_eq!(result.appended_commits, 0);
        let snapshot = pile.snapshot().unwrap();
        assert_eq!(
            ModelCollection::open(&snapshot, result.collection.handle()).unwrap(),
            result.collection,
        );
        let descriptor: Blob<blobencodings::SimpleArchive> =
            snapshot.get(result.collection.handle()).unwrap();
        assert_eq!(descriptor.get_handle(), result.collection.handle());
        drop(snapshot);
        pile.close().unwrap();
    }

    #[test]
    fn migration_is_additive_exact_selectable_resident_and_idempotent() {
        let fixture = legacy_fixture("complete", false);
        let before = std::fs::read(fixture.pile.path()).expect("read legacy prefix");
        let migration_key = key(9);
        let request = LegacyModelMigration {
            model: ModelSelector::Name(LEGACY_MODEL_NAME),
            source: CANONICAL_SOURCE,
            quantization: QUANTIZATION,
            tokenizer_name: Some(CANONICAL_TOKENIZER),
        };

        let mut pile = Pile::open(fixture.pile.path()).expect("open migration pile");
        let first = migrate_legacy_model_main_from_snapshot(
            &mut pile,
            &migration_key,
            &fixture.pins,
            request,
        )
        .expect("migrate legacy graph");
        pile.close().expect("durably close migrated pile");

        assert_eq!(first.legacy_branch, fixture.branch);
        assert_eq!(first.legacy_head, fixture.head);
        assert_eq!(first.model_root, fixture.model_root);
        assert_eq!(first.tokenizer_root, Some(fixture.tokenizer_root));
        assert_eq!(first.legacy_facts, fixture.facts.len());
        assert_eq!(first.selector_facts_added, 2);

        let after_first = std::fs::read(fixture.pile.path()).expect("read migrated pile");
        assert!(
            after_first.starts_with(&before),
            "legacy byte prefix changed"
        );
        assert_eq!(
            read_main_identity(fixture.pile.path(), &fixture.pins),
            (fixture.branch, fixture.head),
            "legacy branch identity/head changed"
        );

        let snapshot = exact_snapshot(fixture.pile.path(), first.collection, first.commit);
        let projection = project_legacy_model_attributes(&fixture.facts);
        assert_eq!(first.aliases_added, projection.aliases_added);
        let mut expected = projection.facts;
        let mut expected_blobs = MemoryBlobStore::new();
        let source = expected_blobs
            .put::<UTF8String, _>(CANONICAL_SOURCE.to_owned())
            .expect("derive expected source handle");
        expected += entity! { ExclusiveId::force_ref(&fixture.model_root) @
            attrs::source: source,
            attrs::quantization: QUANTIZATION,
        }
        .into_facts();
        assert_eq!(
            snapshot.facts(),
            &expected,
            "native facts are not exactly legacy union projected aliases union selector coordinates"
        );
        assert!(snapshot.facts().contains(&fixture.unknown_fact));

        assert_eq!(
            select_model_root(
                snapshot.facts(),
                snapshot.store(),
                ModelSelector::Source {
                    source: CANONICAL_SOURCE,
                    quantization: QUANTIZATION,
                },
            )
            .expect("select migrated model from exact cover"),
            fixture.model_root
        );
        assert_eq!(
            select_tokenizer_root(
                snapshot.facts(),
                snapshot.store(),
                TokenizerSelector::Name(CANONICAL_TOKENIZER),
            )
            .expect("select migrated tokenizer from exact cover"),
            fixture.tokenizer_root
        );
        let attachment: View<str> = snapshot
            .store()
            .get(fixture.attachment)
            .expect("legacy attachment remains resident through exact snapshot");
        assert_eq!(&*attachment, "resident legacy attachment");

        let mut pile = Pile::open(fixture.pile.path()).expect("reopen for idempotence run");
        let second = migrate_legacy_model_main_from_snapshot(
            &mut pile,
            &migration_key,
            &fixture.pins,
            request,
        )
        .expect("rerun migration");
        pile.close().expect("close idempotence run");
        assert_eq!(second.commit, first.commit, "rerun changed signed commit");
        assert_eq!(
            std::fs::read(fixture.pile.path()).expect("read rerun pile"),
            after_first,
            "rerun appended bytes despite identical content-addressed publication"
        );
    }

    /// A legacy pile can hold two weight roots under one `model_name`, which is
    /// exactly what the name selector must refuse. Naming the content address
    /// is how such a pile still migrates. The root selected here is the SECOND
    /// of the two, so a first-match implementation cannot pass by accident.
    #[test]
    fn root_selection_migrates_one_root_of_an_ambiguous_name() {
        let fixture = legacy_fixture("by-root", true);
        let wanted = fixture.duplicate_root.expect("ambiguous fixture");
        assert_ne!(wanted, fixture.model_root);

        let mut pile = Pile::open(fixture.pile.path()).expect("open ambiguous fixture");
        let result = migrate_legacy_model_main_from_snapshot(
            &mut pile,
            &key(9),
            &fixture.pins,
            LegacyModelMigration {
                model: ModelSelector::Root(wanted),
                source: CANONICAL_SOURCE,
                quantization: QUANTIZATION,
                tokenizer_name: None,
            },
        )
        .expect("the content address resolves what the shared name cannot");
        pile.close().expect("close migrated pile");
        assert_eq!(result.model_root, wanted);
        assert_eq!(result.selector_facts_added, 2);

        // The coordinates must land on the requested root and nowhere else, and
        // the shared name must still fail closed afterwards.
        let snapshot = exact_snapshot(fixture.pile.path(), result.collection, result.commit);
        assert_eq!(
            select_model_root(
                snapshot.facts(),
                snapshot.store(),
                ModelSelector::Source {
                    source: CANONICAL_SOURCE,
                    quantization: QUANTIZATION,
                },
            )
            .expect("the migrated root is selectable by its new coordinates"),
            wanted
        );
        let error = select_model_root(
            snapshot.facts(),
            snapshot.store(),
            ModelSelector::Name(LEGACY_MODEL_NAME),
        )
        .expect_err("the shared name is still ambiguous after migration")
        .to_string();
        assert!(error.contains("ambiguous"), "{error}");
    }

    #[test]
    fn ambiguous_legacy_weight_roots_fail_before_publication() {
        let fixture = legacy_fixture("ambiguous", true);
        let before = std::fs::read(fixture.pile.path()).expect("read ambiguous fixture");
        let mut pile = Pile::open(fixture.pile.path()).expect("open ambiguous fixture");
        let error = migrate_legacy_model_main_from_snapshot(
            &mut pile,
            &key(9),
            &fixture.pins,
            LegacyModelMigration {
                model: ModelSelector::Name(LEGACY_MODEL_NAME),
                source: CANONICAL_SOURCE,
                quantization: QUANTIZATION,
                tokenizer_name: None,
            },
        )
        .expect_err("ambiguous roots must fail");
        pile.close().expect("close failed migration pile");
        assert!(error
            .to_string()
            .contains("select exactly one legacy weight root"));
        assert_eq!(
            std::fs::read(fixture.pile.path()).expect("read failed migration pile"),
            before,
            "failed validation mutated the legacy pile"
        );
        assert_eq!(
            read_main_identity(fixture.pile.path(), &fixture.pins),
            (fixture.branch, fixture.head)
        );
    }

    #[test]
    fn duplicate_main_branch_names_fail_before_publication() {
        let fixture = legacy_fixture("duplicate-main", false);
        let mut pile = Pile::open(fixture.pile.path()).expect("open duplicate-main fixture");
        let second_branch = test_id(0x71);
        let second_metadata = write_branch_metadata(
            &mut pile,
            &key(3),
            second_branch,
            &["main"],
            &[fixture.head],
        );
        let mut pins = fixture.pins.clone();
        insert_pin(&mut pins, second_branch, second_metadata);
        let before = std::fs::read(fixture.pile.path()).expect("read ambiguous branch fixture");

        let error = migrate_legacy_model_main_from_snapshot(
            &mut pile,
            &key(9),
            &pins,
            LegacyModelMigration {
                model: ModelSelector::Name(LEGACY_MODEL_NAME),
                source: CANONICAL_SOURCE,
                quantization: QUANTIZATION,
                tokenizer_name: None,
            },
        )
        .expect_err("two active main branches must fail closed");
        assert!(
            error.to_string().contains("2 active branches named 'main'"),
            "{error}"
        );
        pile.close().expect("close duplicate-main fixture");
        assert_eq!(
            std::fs::read(fixture.pile.path()).expect("reread duplicate-main fixture"),
            before,
            "ambiguous branch discovery published native state"
        );
    }

    #[test]
    fn ambiguous_main_head_fails_before_publication() {
        let fixture = legacy_fixture("ambiguous-head", false);
        let mut pile = Pile::open(fixture.pile.path()).expect("open ambiguous-head fixture");
        let second_head = write_legacy_commit(
            &mut pile,
            &key(3),
            Fragment::from_facts_and_blobs(TribleSet::new(), MemoryBlobStore::new()),
            [fixture.head],
        );
        let metadata = write_branch_metadata(
            &mut pile,
            &key(3),
            fixture.branch,
            &["main"],
            &[fixture.head, second_head],
        );
        let mut pins = PinSnapshot::new();
        insert_pin(&mut pins, fixture.branch, metadata);
        let before = std::fs::read(fixture.pile.path()).expect("read ambiguous-head fixture");

        let error = migrate_legacy_model_main_from_snapshot(
            &mut pile,
            &key(9),
            &pins,
            LegacyModelMigration {
                model: ModelSelector::Name(LEGACY_MODEL_NAME),
                source: CANONICAL_SOURCE,
                quantization: QUANTIZATION,
                tokenizer_name: None,
            },
        )
        .expect_err("a branch with two heads must fail closed");
        assert!(error.to_string().contains("ambiguous heads"), "{error}");
        pile.close().expect("close ambiguous-head fixture");
        assert_eq!(
            std::fs::read(fixture.pile.path()).expect("reread ambiguous-head fixture"),
            before,
            "ambiguous head discovery published native state"
        );
    }

    #[test]
    fn audited_personaplex_policy_pins_the_real_commit_shape() {
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.legacy_facts, 4_700);
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.lm, 475);
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.mimi, 318);
        assert_eq!(PERSONAPLEX_MEMBER_POLICY.total, 793);
    }

    #[test]
    fn personaplex_adoption_uses_exact_commit_and_is_a_zero_byte_retry() {
        let fixture = legacy_personaplex_fixture("personaplex-exact", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x71);

        // The later head still contains both weight roots and all three
        // members. Exact legacy fact cardinality is therefore an independent
        // fail-closed guard against accidentally selecting the moving head.
        let before_later_attempt = std::fs::read(fixture.pile.path()).unwrap();
        let mut pile = Pile::open(fixture.pile.path()).unwrap();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.later_commit,
            policy,
        )
        .expect_err("later unrelated head must not pass the audited checkout shape");
        assert!(
            error.to_string().contains("facts, expected exactly"),
            "{error}"
        );
        assert!(
            bundle_collections(&mut pile).is_empty(),
            "failed exact-head validation exposed a bundle COMMIT"
        );
        assert_eq!(
            std::fs::read(fixture.pile.path()).unwrap(),
            before_later_attempt,
            "later-head rejection appended dependencies or a COMMIT"
        );

        let first = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("adopt exact legacy PersonaPlex weight commit");
        assert!(first.published);
        assert_eq!(first.legacy_commit, fixture.weight_commit);
        assert_eq!(first.legacy_lm_root, fixture.lm_root);
        assert_eq!(first.legacy_mimi_root, fixture.mimi_root);
        assert_ne!(first.model_root, fixture.lm_root);
        assert_ne!(first.model_root, fixture.mimi_root);

        let len_after_first = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let repeated = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("repeat exact PersonaPlex adoption");
        assert!(!repeated.published);
        assert_eq!(repeated.commit, first.commit);
        assert_eq!(repeated.model_root, first.model_root);
        assert_eq!(repeated.model_archive_data, first.model_archive_data);
        assert_eq!(
            std::fs::metadata(fixture.pile.path()).unwrap().len(),
            len_after_first,
            "same-policy exact `(root, H)` retry appended bytes"
        );
        pile.close().unwrap();

        let snapshot = exact_bundle_snapshot(fixture.pile.path(), first.collection, first.commit);
        let archive: Blob<blobencodings::SimpleArchive> = snapshot
            .store()
            .get(
                inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(
                    first.model_archive_data,
                ),
            )
            .unwrap();
        let model_facts = TribleSet::try_from_blob(archive).unwrap();
        let projection = project_legacy_model_attributes(&fixture.facts);
        assert!(
            projection
                .facts
                .iter()
                .all(|fact| model_facts.contains(fact)),
            "adoption rewrote or omitted exact legacy facts/ids"
        );
        assert!(
            !model_facts.contains(&fixture.later_fact),
            "exact weight checkout widened to the later current head"
        );
        assert_eq!(
            select_model_root(
                &model_facts,
                snapshot.store(),
                ModelSelector::Source {
                    source: PERSONAPLEX_SOURCE,
                    quantization: mary::persist::QUANTIZATION_NATIVE,
                },
            )
            .unwrap(),
            first.model_root
        );
        assert_eq!(
            select_model_root(
                &model_facts,
                snapshot.store(),
                ModelSelector::Name(PERSONAPLEX_LM_FILE),
            )
            .unwrap(),
            fixture.lm_root
        );
        assert_eq!(
            select_model_root(
                &model_facts,
                snapshot.store(),
                ModelSelector::Name(PERSONAPLEX_MIMI_FILE),
            )
            .unwrap(),
            fixture.mimi_root
        );
        assert!(
            model_facts.iter().all(|fact| {
                fact.e() != &first.model_root || fact.a() != &attrs::model_name.id()
            }),
            "the unified root must not acquire two values for functional model_name"
        );
        let weights = PersonaPlexWeights::from_graph(&model_facts, snapshot.store().clone())
            .expect("load adopted PersonaPlex graph through runtime resolver");
        assert_eq!(weights.root(), first.model_root);
        assert_eq!(weights.count(), policy.total);
    }

    #[test]
    fn exact_personaplex_commit_does_not_require_a_live_legacy_branch() {
        let fixture = legacy_personaplex_fixture("personaplex-detached", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x76);
        let mut pile = Pile::open(fixture.pile.path()).unwrap();

        assert!(
            pile.snapshot_pin_heads()
                .unwrap()
                .iter_ordered()
                .next()
                .is_none(),
            "exact-commit fixture must not manufacture a branch pin"
        );

        let adopted = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("resident exact commit remains adoptable without a branch pin");
        assert!(adopted.published);
        assert_eq!(adopted.legacy_commit, fixture.weight_commit);
        pile.close().unwrap();

        let snapshot =
            exact_bundle_snapshot(fixture.pile.path(), adopted.collection, adopted.commit);
        let archive: Blob<blobencodings::SimpleArchive> = snapshot
            .store()
            .get(
                inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(
                    adopted.model_archive_data,
                ),
            )
            .unwrap();
        let model_facts = TribleSet::try_from_blob(archive).unwrap();
        let weights = PersonaPlexWeights::from_graph(&model_facts, snapshot.store().clone())
            .expect("load branch-independent adopted PersonaPlex graph");
        assert_eq!(weights.root(), adopted.model_root);
        assert_eq!(weights.count(), policy.total);
    }

    #[test]
    fn personaplex_retry_does_not_return_an_unadmitted_token_attestation() {
        let fixture = legacy_personaplex_fixture("personaplex-retry-writer", false, false);
        let policy = personaplex_policy(&fixture);
        let mut keys = [key(0x79), key(0x7a)];
        keys.sort_by_key(|key| key.verifying_key().to_bytes());
        let [unadmitted_key, migration_key] = keys;
        let mut pile = Pile::open(fixture.pile.path()).unwrap();
        let first = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("adopt exact legacy PersonaPlex weight commit");
        assert!(first.published);

        let unadmitted = CollectionCommit::sign(
            &unadmitted_key,
            first.collection.handle(),
            first.commit.data(),
            first.commit.metadata(),
        );
        pile.insert(CollectionRecord::Commit(unadmitted)).unwrap();
        let observation = pile.snapshot().unwrap();
        let support = first.collection.admitted(&observation).unwrap();
        assert_eq!(support.len(), 1);
        assert_eq!(
            member_commits(&support, &observation).unwrap(),
            vec![unadmitted, first.commit],
            "the unadmitted attestation must sort before the legitimate retry result"
        );
        assert!(!first
            .collection
            .writer_is_admitted(&observation, unadmitted_key.verifying_key())
            .unwrap());
        drop(observation);

        let before = std::fs::read(fixture.pile.path()).unwrap();
        let repeated = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("repeat adoption with an extra unadmitted token attestation");
        assert!(!repeated.published);
        assert_eq!(repeated.commit, first.commit);
        assert_eq!(std::fs::read(fixture.pile.path()).unwrap(), before);
        pile.close().unwrap();
    }

    #[test]
    fn personaplex_count_and_overlap_fail_before_any_publication() {
        let count_fixture = legacy_personaplex_fixture("personaplex-count", false, false);
        let mut wrong_count = personaplex_policy(&count_fixture);
        wrong_count.lm += 1;
        let before = std::fs::read(count_fixture.pile.path()).unwrap();
        let mut pile = Pile::open(count_fixture.pile.path()).unwrap();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &key(0x72),
            count_fixture.weight_commit,
            wrong_count,
        )
        .expect_err("wrong audited LM count must fail");
        assert!(error.to_string().contains("members, expected"), "{error}");
        assert!(bundle_collections(&mut pile).is_empty());
        pile.close().unwrap();
        assert_eq!(std::fs::read(count_fixture.pile.path()).unwrap(), before);

        let overlap_fixture = legacy_personaplex_fixture("personaplex-overlap", true, false);
        let policy = personaplex_policy(&overlap_fixture);
        let before = std::fs::read(overlap_fixture.pile.path()).unwrap();
        let mut pile = Pile::open(overlap_fixture.pile.path()).unwrap();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &key(0x73),
            overlap_fixture.weight_commit,
            policy,
        )
        .expect_err("shared LM/Mimi member must fail");
        assert!(error.to_string().contains("share member"), "{error}");
        assert!(bundle_collections(&mut pile).is_empty());
        pile.close().unwrap();
        assert_eq!(std::fs::read(overlap_fixture.pile.path()).unwrap(), before);
    }

    #[test]
    fn malformed_personaplex_fails_after_staging_without_a_bundle_commit() {
        let fixture = legacy_personaplex_fixture("personaplex-malformed", false, true);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x77);
        let before = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let mut pile = Pile::open(fixture.pile.path()).unwrap();

        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect_err("duplicate tensor names must fail runtime validation");
        assert!(
            format!("{error:#}").contains("duplicate tensor name"),
            "{error:#}"
        );
        assert!(
            std::fs::metadata(fixture.pile.path()).unwrap().len() > before,
            "the falsifier must cross the dependency-staging boundary"
        );
        assert!(
            bundle_collections(&mut pile).is_empty(),
            "failed staged validation finalized a bundle COMMIT"
        );
        pile.close().unwrap();
    }

    #[test]
    fn personaplex_conflict_is_detected_before_staging_dependencies() {
        let fixture = legacy_personaplex_fixture("personaplex-conflict", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x74);
        let conflicting = conflicting_personaplex_fragment();
        let conflicting_root = conflicting.root().unwrap();
        let mut pile = Pile::open(fixture.pile.path()).unwrap();
        let existing =
            publish_model_bundle_fragment(&mut pile, &migration_key, conflicting_root, conflicting)
                .expect("publish conflicting same-policy PersonaPlex bundle");
        let before = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect_err("a different authoritative PersonaPlex bundle must conflict");
        assert!(
            error
                .to_string()
                .contains("different PersonaPlex bundle is already authoritative"),
            "{error}"
        );
        assert_eq!(
            std::fs::metadata(fixture.pile.path()).unwrap().len(),
            before,
            "conflict staged dependencies before failing"
        );
        let collection = model_bundle_collection_or_create(&mut pile, &migration_key).unwrap();
        let store = pile.snapshot().unwrap();
        let cover = collection.admitted(&store).unwrap();
        let commits: Vec<_> = member_commits(&cover, &store)
            .unwrap()
            .into_iter()
            .filter(|commit| {
                let writer = VerifyingKey::from_bytes(&commit.public_key().raw).unwrap();
                collection.writer_is_admitted(&store, writer).unwrap()
            })
            .collect();
        assert_eq!(cover.len(), 1);
        assert_eq!(commits, vec![existing]);
        drop(store);
        pile.close().unwrap();
    }

    #[test]
    fn same_personaplex_root_with_different_archive_conflicts_before_staging() {
        let fixture = legacy_personaplex_fixture("personaplex-same-root-conflict", false, false);
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x78);
        let mut pile = Pile::open(fixture.pile.path()).unwrap();

        pile.refresh().unwrap();
        let reader = pile.snapshot().unwrap();
        let legacy = checkout_legacy_commit_ancestors(&reader, fixture.weight_commit).unwrap();
        let mut conflicting =
            prepare_legacy_personaplex_candidate(legacy, &reader, policy).unwrap();
        drop(reader);

        // Preserve the exact intrinsic PersonaPlex root while changing H with
        // one unrelated, valid row. This distinguishes pair equality from an
        // incorrect `same root || same archive` retry test.
        let unrelated = Trible::force(
            &test_id(0x79),
            &test_id(0x7a),
            &inlineencodings::GenId::inline_from(test_id(0x7b)),
        );
        conflicting.fragment += Fragment::from_facts_and_blobs(
            std::iter::once(unrelated).collect(),
            MemoryBlobStore::new(),
        );
        assert_eq!(conflicting.fragment.root(), Some(conflicting.model_root));
        let existing = publish_model_bundle_fragment(
            &mut pile,
            &migration_key,
            conflicting.model_root,
            conflicting.fragment,
        )
        .expect("publish same-root different-H authority");

        let before = std::fs::metadata(fixture.pile.path()).unwrap().len();
        let error = adopt_legacy_personaplex_bundle_with_policy(
            &mut pile,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect_err("same root with a different H must conflict");
        assert!(
            error
                .to_string()
                .contains("different PersonaPlex bundle is already authoritative"),
            "{error}"
        );
        assert_eq!(
            std::fs::metadata(fixture.pile.path()).unwrap().len(),
            before,
            "same-root different-H conflict staged dependencies"
        );
        let collection = model_bundle_collection_or_create(&mut pile, &migration_key).unwrap();
        let store = pile.snapshot().unwrap();
        let cover = collection.admitted(&store).unwrap();
        let commits: Vec<_> = member_commits(&cover, &store)
            .unwrap()
            .into_iter()
            .filter(|commit| {
                let writer = VerifyingKey::from_bytes(&commit.public_key().raw).unwrap();
                collection.writer_is_admitted(&store, writer).unwrap()
            })
            .collect();
        assert_eq!(cover.len(), 1);
        assert_eq!(commits, vec![existing]);
        drop(store);
        pile.close().unwrap();
    }

    #[test]
    fn retained_rewrite_reaches_legacy_tensor_blobs_only_through_bundle_commit() {
        let fixture = legacy_personaplex_fixture("personaplex-retention", false, false);
        let destination = TempPilePath::new("personaplex-retained");
        let policy = personaplex_policy(&fixture);
        let migration_key = key(0x75);
        let mut source = Pile::open(fixture.pile.path()).unwrap();
        let adopted = adopt_legacy_personaplex_bundle_with_policy(
            &mut source,
            &migration_key,
            fixture.weight_commit,
            policy,
        )
        .expect("adopt bundle before retained rewrite");

        assert_eq!(
            source.snapshot_pin_heads().unwrap().iter_ordered().count(),
            0,
            "a legacy pin would confound native bundle-retention coverage"
        );

        let mut retained = Pile::open(destination.path()).unwrap();
        let source_store = source.snapshot().unwrap();
        let support = local_model_support(&source_store, adopted.collection).unwrap();
        drop(source_store);
        source
            .rewrite_retained_into(
                &mut retained,
                &RetentionRoots::new(),
                WantRewritePolicy::Drop,
            )
            .expect("rewrite native bundle closure");
        source.close().unwrap();

        let retained_store = retained.snapshot().unwrap();
        let snapshot = snapshot_model_bundle_collection_exact(&retained_store, &support)
            .expect("materialize retained exact bundle");
        let token_archive: Blob<blobencodings::SimpleArchive> = snapshot
            .store()
            .get(
                inlineencodings::Handle::<blobencodings::SimpleArchive>::from_hash(
                    adopted.commit.data(),
                ),
            )
            .expect("COMMIT retained its one-row token T");
        let token = TribleSet::try_from_blob(token_archive).unwrap();
        assert_eq!(&token, snapshot.facts());
        assert_eq!(token.len(), 1);
        let token_row = token.iter().next().unwrap();
        assert_eq!(token_row.e(), &adopted.model_root);
        assert_eq!(token_row.a(), &metadata::archive.id());
        let model_archive = *token_row.v::<inlineencodings::Handle<blobencodings::SimpleArchive>>();
        assert_eq!(
            inlineencodings::Handle::<blobencodings::SimpleArchive>::to_hash(model_archive),
            adopted.model_archive_data
        );
        let archive: Blob<blobencodings::SimpleArchive> = snapshot
            .store()
            .get(model_archive)
            .expect("T retained the exact model archive H");
        let facts = TribleSet::try_from_blob(archive).unwrap();
        assert!(
            fixture.facts.iter().all(|fact| facts.contains(fact)),
            "retained H omitted original historical rows"
        );
        for (index, ((leaf_id, data), shape)) in fixture
            .leaf_ids
            .iter()
            .zip(&fixture.data)
            .zip(&fixture.shapes)
            .enumerate()
        {
            let payload: View<[f32]> = snapshot.store().get(*data).unwrap();
            let dimensions: View<[u64]> = snapshot.store().get(*shape).unwrap();
            assert_eq!(&*payload, &[index as f32 + 1.0]);
            assert_eq!(&*dimensions, &[1]);
            let leaf = mary::leaf::resolve(&facts, snapshot.store(), *leaf_id)
                .unwrap()
                .expect("legacy two-blob leaf remains resolvable");
            assert_eq!(leaf.dims(), &[1]);
            assert_eq!(&*leaf.view_f32().unwrap(), &[index as f32 + 1.0]);
        }
        assert!(
            snapshot.store().get::<Bytes, _>(fixture.orphan).is_err(),
            "unrelated later-commit attachment survived without an ownership path"
        );
        drop(snapshot);
        retained.close().unwrap();
    }
}
