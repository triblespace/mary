#!/bin/bash
# gb10-lock.sh -- ONE advisory lock per GB10 box, taken by EVERY agent that runs
# on them, so two measurements cannot overlap.
#
# WHY THIS EXISTS. 2026-08-27, 01:5x: two agents ran decode processes on the same
# pair of boxes at the same time. Each admits ~101 GiB on a 121 GiB part, so the
# second one did not make the first slower -- it OOM-killed it. The victim's head
# process took SIGKILL (rc=137) on rep 2 and its own settle check printed "still
# 19/114 GiB head... this rep may be paging". Neither agent did anything wrong:
# both checked the box was idle before starting, and both were right at the
# moment they looked. An idle CHECK is not a RESERVATION, and the window between
# looking and launching is exactly long enough to lose a night's measurement.
#
# scripts/frontier-bench.sh already has a lock of this shape, but it is named for
# frontier and only frontier takes it, so it protects frontier from frontier and
# nothing from anything else. This is the same mechanism with the scope it needed.
#
# THE MECHANISM. `mkdir` is atomic on every filesystem that matters: it either
# creates the directory or fails, and two callers cannot both win. The directory
# holds an `info` file naming host, tag, start time and last heartbeat, so a
# holder can be identified rather than merely detected.
#
# BREAKING A STALE LOCK. Age is all we have. GB10_LOCK_TIMEOUT_S (default
# 5400 s / 90 min) applies, chosen to clear a cold build of mary+burn+cubecl
# plus ~18 min of measurement, so a healthy run is never broken into while one
# crash costs at most one slot rather than the night.
#
# THERE IS DELIBERATELY NO PID-LIVENESS TEST, and this header used to promise
# one: "the lock records a pid AND the host that pid lives on; if that host is
# the box we are on and the pid is gone, the lock is broken IMMEDIATELY". That
# was removed, because the pid it recorded was the transient ssh shell that took
# the lock, which exits the instant the take returns -- so every lock was born
# dead and the very next caller broke it. The take path below explains it again
# at the point where the test used to live.
#
# The promise outlived the code. On 2026-09-17 a reader compared this paragraph
# against a live info file, found no pid= line, and reported the ABSENCE as the
# bug -- reasoning correctly from a description that was no longer true, and
# very nearly restoring the exact test whose removal is documented forty lines
# further down. A stale comment does not merely fail to help; it recruits people
# into undoing the fix.
#
# TAKING THE LOCK OBLIGES YOU TO KEEP BEATING. Silence is indistinguishable
# from death, by construction: a holder doing one long uninterrupted stretch
# without calling `refresh` is treated as crashed once the timeout elapses and
# its reservation is broken WHILE IT IS STILL RUNNING. A 7-rep two-node run is
# ~18 min and a cold build can be 30-40, so a naive holder that takes the lock
# once and works is inside the 90-minute window today, but not by much. The
# second layer catches the consequence -- a breaker's idle gate immediately sees
# the true holder's processes and refuses -- so the failure mode is a lost
# reservation and a wasted slot rather than an OOM. Beat anyway.
#
# USAGE, and note every call passes paths in a FILE rather than on a command line:
#   gb10-lock.sh take <host> <tag>     -> rc 0 took it, rc 3 someone else holds it
#   gb10-lock.sh release <host> <tag>  -> releases only if <tag> holds it
#   gb10-lock.sh check <host>          -> prints the holder, rc 0 free, rc 3 held,
#                                        rc 4 UNREACHABLE (state unknown -- never
#                                        treat this as free OR as held)
#
# BUILDS TAKE THE LOCK TOO. A cargo build is not a measurement, but it is 20
# cores of CPU contention on a unified-memory part, so it perturbs a neighbour's
# decode measurement just as surely as a second decode process would. The busy
# gates already refuse while one is running -- `BOX_BUSY_PATTERN` matches on
# `pgrep -f`, and `--bin inkling_forward` sits in a cargo command line, so a
# build is caught by a pattern written for measurements. THAT IS THE RIGHT
# ANSWER ARRIVED AT BY ACCIDENT, and it must not be "fixed" by narrowing the
# pattern to `pgrep -x`: that would make the gate precise and wrong.
#
# The consequence is that the boxes are unavailable during other agents' builds
# whether or not anyone reserved them, which makes the queue dishonest -- a
# waiter cannot see why it is waiting. So: reserve for the BUILD as well as the
# reps, exactly as scripts/frontier-bench.sh does, and the queue then reflects
# what the boxes are actually spending their time on.
#
# The binary cannot be built off-box: it targets aarch64 Linux with CUDA and the
# only machines that are, are the measurement machines. Building elsewhere is not
# an available answer, which is why reserving is.
#
# IF YOU CAN SEE THE BOX, LOOK BEFORE YOU BREAK A STALE LOCK. This script reads
# silence as death because it cannot see the holder -- that is the right rule for
# the mechanism, and the wrong rule for a caller with ssh. A holder doing one long
# uninterrupted stretch without calling `refresh` goes stale WHILE STILL MEASURING,
# and taking its box destroys a live reservation. The taker's own idle gate then
# refuses and hands the box straight back, so the outcome is not an OOM -- it is a
# reservation deleted and a slot wasted for nothing. Two ssh round trips per poll
# removes the whole case. It also covers the agent who has not adopted the lock at
# all: a box busy with an UNLOCKED run is not one to reserve either.
# (scripts/frontier-bench.sh implements this as `boxes_look_idle && take_both`,
# and the rule is theirs.)
#
# TAKING BOTH BOXES: BOTH OR NEITHER, AND NEVER WAIT WHILE HOLDING HALF.
# A two-node run needs spark and spark2 together. If you take one, fail to get
# the other, and then WAIT while still holding the first, you deadlock against
# any other two-node agent doing the same thing in the opposite order -- A holds
# spark2 waiting for spark, B holds spark waiting for spark2, and neither ever
# yields. Nothing in this script prevents that; it is a property of how you call
# it. So: if the second take fails, RELEASE THE FIRST before you sleep, and ask
# for both again from scratch on the next attempt. Holding half of what you need
# while you wait is the one usage that turns a working lock into a hang.
# (Found by the frontier harness, which hit the half-held state and only escaped
# it because that version exited instead of waiting.)
#
# KNOWN GAP: THE TIMEOUT IS SIZED FOR THE WRONG FAILURE. 5400 s was chosen so a
# healthy holder is never broken into -- a cold build plus reps -- which is the
# right size for a SLOW holder and the wrong size for an ABSENT one. Absent is
# what actually happens: 2026-08-27, a holder beat twice, finished its run, exited
# without releasing, and left two idle GB10s reserved for the remaining 68 minutes
# of its timeout while four agents queued behind it.
#
# THE COST OF NOT HAVING IT, MEASURED ONCE: 21 box-minutes. That episode ran
# from the holder's last beat to the moment the next agent took the boxes, on a
# night with five agents queued, and nothing in the mechanism could have
# shortened it. The conjunction below would have caught it in under a minute.
#
# The better test is a CONJUNCTION rather than elapsed time: if the boxes are idle
# AND the beat has stopped, the holder is done, and that pair is far more
# informative than age alone. A holder mid-build is silent but its box is busy; a
# holder that has exited is silent and its box is idle. scripts/frontier-bench.sh
# already computes half of it in `boxes_look_idle`.
#
# NOT IMPLEMENTED HERE, deliberately. Changing what "abandoned" means while five
# agents are mid-flight is the move that turns a coordination bug into an
# incident, and this gap costs idle machines rather than corrupted measurements.
# The mitigation that IS available today costs nothing and belongs in every
# caller: RELEASE FROM A TRAP on every exit path, including the error and
# interrupt paths. Taking is deliberate, beating is printed as an obligation, and
# releasing is the step a crash skips -- which is why silence after work looks
# exactly like silence during it.
#
# WHAT THIS DOES NOT DO: THERE IS NO QUEUE. It is mutual exclusion and nothing
# more. A caller refused with rc 3 is not remembered, is not owed the next slot,
# and will not be handed anything when the lock frees -- it has to come back and
# ask again. Two consequences, both seen within an hour of this landing:
#
#   - A waiter that gives up on the first refusal simply does not run. If your
#     work matters, POLL with a bounded backoff and a long ceiling, and report
#     "never got a slot" rather than reporting nothing.
#   - With several agents and long holds, an unlucky waiter can STARVE. Nothing
#     here prevents it. If that starts happening, the fix is a real queue (a
#     ticket file, taken in order), not a longer timeout and not politeness.
#
#     IT HAPPENED, AND IT IS MEASURED. 2026-08-27: one waiter polled every
#     180 s for two hours -- 40 cycles -- and found ZERO free windows. 28
#     refusals came from the busy gate and 12 from the lock, across three
#     holders running back to back with no gap between them. Nobody did
#     anything wrong: each holder took the boxes the moment they were free,
#     correctly, and the waiter never won a race it was not trying to enter.
#
#     The mechanism is that RELEASE AND RE-TAKE ARE ADJACENT. A holder
#     releases, the next agent's poll lands within seconds, and the free
#     window never survives to the waiter's next tick. Polling faster does
#     not fix it -- it converts a queue into a lottery with more entrants.
#     What is missing is a TICKET: a record of 'asked at T' so the box goes
#     to the longest waiter rather than to whoever's timer fires first.
#
#     AND IT WILL RECUR, because the density is a CONSEQUENCE OF SUCCESS.
#     Those three agents finished back to back precisely BECAUSE the lock
#     made them serialise cleanly -- the mechanism that prevents the OOM is
#     the mechanism that produces the convoy. So starvation is not an
#     anomaly to be waited out; it is what a busy night looks like when this
#     file is doing its job, and it will happen again every time it does.
#
#     Until that exists, the queue is a PERSON. An arbiter who knows the
#     ordering can hold a released slot for the starving waiter, which is
#     what happened this night. That is not a substitute for the ticket; it
#     is what you do while the ticket does not exist.
#
#     AND THE TWO FAIL DIFFERENTLY, which is the part worth carrying: a
#     ticket fails by going STALE, a person fails by going to SLEEP. Both
#     are recoverable and only one of them announces itself. A stale ticket
#     sits there to be found; an absent arbiter leaves no trace at all, and
#     the next person to read this at 3am will BE the failure mode.
#
# It is deliberately this simple because the failure it was built for -- two
# runs overlapping and OOM-killing each other -- needs only exclusion, and a
# queue that is wrong is worse than no queue. Revisit when starvation is
# observed, not before.
#
# AND WHEN YOU DO KILL BY PID, RE-VERIFY THE PID IN THE SAME ROUND TRIP.
# "Kill by PID, never by pattern" is not the whole rule, because it treats a PID
# as a stable identifier and it is only stable while the process lives. On a box
# that forks constantly, a PID copied out of a report written minutes ago can name
# something else entirely by the time the signal lands. So the two failures are
# symmetric and both are live here: killing by PATTERN is unsafe because it can
# match a neighbour, and killing by a STALE PID is unsafe because it can BECOME
# one. Check /proc/<pid>/cmdline still says what you expect, in the same ssh call
# that sends the signal -- not in a previous one, and not from a message someone
# sent you.
#
# 2026-08-27: two unkillable waiters were reported and I was about to kill them on
# the strength of the report. Verifying first showed both had already been reaped
# by their owner, so the signals would have gone to whatever had inherited those
# numbers on a box that had forked thousands of times since.
#
# NEVER `pkill -f` ON A SHARED BOX, and never kill a lock holder's processes to
# take a lock. If a box is held, WAIT or exit cleanly and say so. The lock tells
# you who to ask, not who to kill.
set -uo pipefail

ACTION=${1:-}; HOST=${2:-}; TAG=${3:-}
: "${GB10_LOCK_TIMEOUT_S:=5400}"

usage() { sed -n '2,40p' "$0"; exit 2; }
[ -z "$ACTION" ] && usage
[ -z "$HOST" ] && usage
case "$ACTION" in take|release) [ -z "$TAG" ] && usage ;; esac

# The payload is a QUOTED heredoc: nothing in it expands locally. Everything it
# needs arrives as a positional argument. An unquoted heredoc would run $$,
# $(date) and $(hostname) on THIS machine and write a lock describing the wrong
# host -- which is worse than no lock, because it looks like one.
# NOTE: no `ssh -n` here. -n redirects stdin from /dev/null, which silently
# eats the heredoc and makes every call return nothing at all -- an empty
# answer that reads as "free" is the most dangerous possible failure for a
# lock, so it is called out rather than just avoided.
timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  "bash -s -- '$ACTION' '$TAG' '$GB10_LOCK_TIMEOUT_S' '$HOST'" <<'REMOTE'
set -uo pipefail
# HOSTALIAS is the ssh alias the CALLER used. It is passed in purely so the
# take message can print a command that actually runs: `refresh` takes
# <host> <tag> like every other verb, and a message that says `refresh $TAG`
# gets followed literally, fails with "Could not resolve hostname <tag>",
# and then the lock goes stale mid-measurement while another agent takes the
# box. Cost one round trip on 2026-08-30; the silent-stale outcome is worse.
ACTION=$1; TAG=$2; TIMEOUT=$3; HOSTALIAS=${4:-<host>}
LOCKD="$HOME/gb10/box.lock.d"
INFO="$LOCKD/info"
ME=$(hostname)
NOW=$(date -u +%s)

write_info() {
  printf 'host=%s\ntag=%s\nstart=%s\nbeat=%s\n' "$ME" "$TAG" "$NOW" "$NOW" > "$INFO"
}

case "$ACTION" in
  take)
    mkdir -p "$HOME/gb10"
    if mkdir "$LOCKD" 2>/dev/null; then write_info
      echo "TAKEN $TAG -- you MUST call 'refresh $HOSTALIAS $TAG' at least every ${TIMEOUT}s"
      echo "  or this lock goes stale and another agent may take the box while you run."
      exit 0; fi
    [ -f "$INFO" ] || { echo "HELD by an unidentified holder"; exit 3; }
    hhost=$(sed -n 's/^host=//p' "$INFO")
    htag=$(sed -n 's/^tag=//p' "$INFO");  hst=$(sed -n 's/^beat=//p' "$INFO")
    age=$(( NOW - ${hst:-0} ))
    # TAKING A LOCK YOU ALREADY HOLD SUCCEEDS. `mkdir` fails when the directory
    # exists, so without this a caller is refused BY ITSELF -- which breaks the
    # two cases that matter most for unattended work: a crashed run relaunched
    # with the same tag (exactly what a restartable job does), and a third party
    # taking it on your behalf. It is safe because it succeeds only when the
    # holder IS you, so it can never put two tags on one box. Beat while we are
    # here, since a re-take is proof of life.
    if [ "$htag" = "$TAG" ]; then
      sed -i "s/^beat=.*/beat=$NOW/" "$INFO" 2>/dev/null
      echo "TAKEN $TAG (already held by you; beat refreshed)"; exit 0
    fi
    # THERE IS DELIBERATELY NO PID-LIVENESS TEST, and the reason is worth keeping.
    # The first version had one: "a dead pid on this box is a dead lock, no
    # timeout needed", which reads as the more exact answer. It is the more exact
    # answer to the WRONG QUESTION. The pid it recorded was the transient ssh
    # shell that took the lock, which exits the instant the take returns -- so
    # every lock was born dead and the very next caller broke it. Caught by this
    # script's own self-test, where `take b` cheerfully broke `take a` one second
    # later and reported it as a repair. Liveness here is a HEARTBEAT, not a pid:
    # a holder doing long work calls `refresh` to prove it is still there.
    if [ "$age" -gt "$TIMEOUT" ]; then
      rm -rf "$LOCKD"
      if mkdir "$LOCKD" 2>/dev/null; then write_info
        echo "TAKEN $TAG (broke a stale lock: $htag silent ${age}s > ${TIMEOUT}s)"; exit 0; fi
    fi
    echo "HELD by tag=$htag on $hhost, last beat ${age}s ago -- WAIT or exit, do not kill"
    exit 3
    ;;
  refresh)
    # A long run proves it is alive. Cheap, and it lets the timeout be short
    # enough that a crash costs one slot instead of the night.
    [ -d "$LOCKD" ] || { echo "NOT HELD"; exit 3; }
    htag=$(sed -n 's/^tag=//p' "$INFO" 2>/dev/null)
    [ "$htag" = "$TAG" ] || { echo "REFUSING: held by $htag, not $TAG"; exit 3; }
    sed -i "s/^beat=.*/beat=$NOW/" "$INFO" && echo "BEAT $TAG"; exit 0
    ;;
  release)
    [ -d "$LOCKD" ] || { echo "NOT HELD"; exit 0; }
    htag=$(sed -n 's/^tag=//p' "$INFO" 2>/dev/null)
    if [ "$htag" = "$TAG" ]; then rm -rf "$LOCKD"; echo "RELEASED $TAG"; exit 0; fi
    echo "REFUSING: held by $htag, not $TAG"; exit 3
    ;;
  check)
    [ -d "$LOCKD" ] || { echo "FREE"; exit 0; }
    tr '\n' ' ' < "$INFO" 2>/dev/null; echo
    exit 3
    ;;
  *) echo "usage: gb10-lock.sh take|refresh|release|check <host> [tag]"; exit 2 ;;
esac
REMOTE
rc=$?
# UNREACHABLE IS NOT FREE, AND IT IS NOT HELD. The remote payload exits only
# 0 (free / action succeeded), 2 (usage) or 3 (held), so any other code came
# from ssh or timeout, i.e. we never learned the lock's state at all. Before
# 2026-08-30 that surfaced as an ssh error on stderr and a non-matching stdout,
# which every "wait until it says FREE" loop reads as "still held" -- so a
# waiter sat on a box that DHCP had moved for 332 minutes, silently, because a
# box you cannot reach and a box someone is using look identical from inside
# the loop. Give it its own word and its own exit code so a waiter can say
# which one it is.
case "$rc" in
  0|2|3) exit "$rc" ;;
  124) echo "UNREACHABLE $HOST: ssh timed out (60s)." ;;
  255) echo "UNREACHABLE $HOST: ssh could not connect or authenticate." ;;
  *)   echo "UNREACHABLE $HOST: ssh exited $rc." ;;
esac
echo "  The box may be down, or DHCP may have moved it. Both boxes stay reachable"
echo "  over ZeroTier regardless: ssh spark2-zt (=sky) / spark-zt (=stars), and"
echo "  \`ip -br addr show enP7s7\` there prints the current wired address."
exit 4
