# How to push this version to GitHub

This version (v2.1) renames the method to **GradFusion** and keeps only what belongs to it.
Old files are removed, so the steps below also delete them from the repository. Delete this
file afterwards if you like; it is instructions for you, not part of the project.

In your Codespace (GitHub → your repository → **Code** → **Codespaces**), drag
`GradFusion-v2.1.zip` onto the file list, then run in the terminal:

```bash
git pull                                      # start from the latest version
git tag before-gradfusion                     # bookmark what is on GitHub now
git push origin before-gradfusion

git ls-files -z | xargs -0 rm -f              # remove every tracked file
unzip -q -o GradFusion-v2.1.zip -d /tmp/new   # unpack outside the repository
cp -a /tmp/new/GradFusion/. .                 # copy the new version, hidden files too
rm -rf /tmp/new GradFusion-v2.1.zip

python -m pytest tests/ -q                    # should print: 36 passed
git add -A                                    # stages new, changed and deleted files
git commit -m "v2.1: rename to GradFusion, add architecture figures"
git push
```

Afterwards the **Actions** tab runs the tests and turns the README badge green.

Nothing is lost: the old version stays in the history, and the tag `before-gradfusion` takes
you back to it (`git checkout before-gradfusion`).

**Optional: rename the repository.** The repository is still called `AdaDiag`. To rename it,
open **Settings** on GitHub and change the repository name to `GradFusion`. GitHub redirects
the old link. Then replace `ShehreyarRashid/AdaDiag` with `ShehreyarRashid/GradFusion` in
`README.md`, `CITATION.cff` and `pyproject.toml`.
