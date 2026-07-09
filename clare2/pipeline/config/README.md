# Remote Corpus Sync Setup

`corpus_sources.yml` in this directory lists the developer machines that the
nightly `corpus_sync` job (runs at 21:30 UTC, before distillation) pulls
captured sessions from. See the comments in that file for the entry format.

The easiest way to manage subscriptions is
`clare2/scripts/clare2-corpus-manage.sh`, which handles keypair generation,
remote key installation, host-key pinning, and manual syncs for you:

```bash
clare2/scripts/clare2-corpus-manage.sh list
clare2/scripts/clare2-corpus-manage.sh subscribe jketreno@dev-laptop.example.com
clare2/scripts/clare2-corpus-manage.sh sync
clare2/scripts/clare2-corpus-manage.sh unsubscribe jketreno@dev-laptop.example.com
```

## What `subscribe` does

1. Generates the CLARE2 corpus sync keypair on first use, if one doesn't
   already exist (`./secrets/clare2_corpus_sync_key` by default — the private
   half is mounted into `clare2-policy` as the `clare2_corpus_sync_key`
   Docker secret, `CLARE2_CORPUS_SYNC_KEY_FILE` in `docker-compose.yml`).
2. Checks whether the sync key already has restricted rsync access to the
   target. If not, connects to the target over SSH (may prompt for that
   user's password) and appends a line to their `~/.ssh/authorized_keys`
   restricted to read-only rsync of their own corpus `sessions/` subtree:

   ```
   command="rrsync -ro ~/.config/clare/corpus/sessions",restrict,no-agent-forwarding,no-X11-forwarding,no-port-forwarding ssh-ed25519 AAAA...
   ```

   `rrsync` ships with the `rsync` package. The forced command confines the
   connection to that one directory — it cannot read anything else on the
   remote host even if the private key were somehow reused elsewhere.
3. Pins the host's key fingerprint via `ssh-keyscan` so sync never falls
   back to unpinned host-key checking, and writes the entry to
   `corpus_sources.yml`.

## What `sync` does

Reads every entry in `corpus_sources.yml` and rsyncs each host's `sessions/`
subtree into `$CLARE2_CORPUS_ROOT/sessions/` locally, using the pinned
`host_key` for each host (never falls back to unpinned host-key checking).
This is the same operation the nightly `corpus_sync` job runs — `sync` just
lets you run it on demand from the command line, without needing
`clare2-policy` up. A single unreachable host is reported but does not stop
the others from syncing.

You can also trigger the nightly job's own HTTP endpoint if `clare2-policy`
is already running (e.g. to exercise the same code path and metrics as the
scheduled run):

```bash
curl -X POST -H "Authorization: Bearer $CLARE2_OPERATOR_TOKEN" \
  http://127.0.0.1:${CLARE2_PROXY_PORT:-8000}/corpus/sync
```

Then check `$CLARE2_CORPUS_ROOT/meta/corpus_sync_status.json` for the
per-host outcome.

Only `sessions/` is ever pulled from a remote host. Episodes, themes, and
training data are generated locally and are never sourced from a remote.
