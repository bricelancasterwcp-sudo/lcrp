# Push this tree

Empty repo: https://github.com/bricelancasterwcp-sudo/lcrp

```bash
tar -xzf lcrp-phase0.tar.gz
cd lcrp
git init
git remote add origin git@github.com:bricelancasterwcp-sudo/lcrp.git
git fetch origin
git checkout -b docs/phase0
# keep remote README history if desired:
# git pull origin main --allow-unrelated-histories
git add .
git commit -m "docs: Phase 0 LCRP architecture, eval plan, harness, gates"
git push -u origin docs/phase0
gh pr create --base main --head docs/phase0 \
  --title "Phase 0: LCRP design pack" \
  --body "Architecture, bloomery accounting, assay eval plan, sensorium harness, borrowed laws, gates, and scaffold."
```

Or hand the bot a fine-grained PAT with Contents + PRs write on `lcrp` and ask it to push.
