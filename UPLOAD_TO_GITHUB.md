# How to push this version to GitHub

This version (v2.2) is a code review of v2.1: bug fixes, input checks, less duplicated code,
an optional fast mode and 18 new tests. Results are unchanged (see `docs/CODE_REVIEW.md`).
Delete this file afterwards if you like; it is instructions for you, not part of the project.

In your Codespace (GitHub → your repository → **Code** → **Codespaces**), drag
`GradFusion-v2.2.zip` onto the file list, then run in the terminal:

```bash
git pull
git ls-files -z | xargs -0 rm -f              # remove old files
unzip -q -o GradFusion-v2.2.zip -d /tmp/new
cp -a /tmp/new/GradFusion/. .
rm -rf /tmp/new GradFusion-v2.2.zip
pip install -q -r requirements.txt
python -m pytest tests/ -q                    # expect 54 passed, 1 skipped
git add -A
git commit -m "v2.2: code review, input checks, optional fast mode"
git push
```

Afterwards the **Actions** tab runs the tests and turns the README badge green.

**Optional: rename the repository.** The repository is still called `AdaDiag`. To rename it,
open **Settings** on GitHub and change the repository name to `GradFusion`. GitHub redirects
the old link. Then replace `ShehreyarRashid/AdaDiag` with `ShehreyarRashid/GradFusion` in
`README.md`, `CITATION.cff` and `pyproject.toml`.
