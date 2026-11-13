# How to push this version to GitHub

This version (v2.0) keeps only AdaDiag++ and what belongs to it. Many old files are removed,
so the steps below also delete them from the repository. Delete this file afterwards if you
like; it is instructions for you, not part of the project.

In your Codespace (GitHub → your repository → **Code** → **Codespaces**), drag
`AdaDiag-v2.0.zip` onto the file list, then run in the terminal:

```bash
git pull                                      # start from the latest version
git tag before-cleanup                        # bookmark the full research version
git push origin before-cleanup

git ls-files -z | xargs -0 rm -f              # remove every tracked file
unzip -q -o AdaDiag-v2.0.zip -d /tmp/new      # unpack outside the repository
cp -a /tmp/new/AdaDiag/. .                    # copy the new version, hidden files too
rm -rf /tmp/new AdaDiag-v2.0.zip

python -m pytest tests/ -q                    # should print: 36 passed
git add -A                                    # stages new, changed and deleted files
git commit -m "v2.0: keep AdaDiag++ only"
git push
```

Afterwards the **Actions** tab runs the tests and turns the README badge green.

Nothing is lost: the old version stays in the history, and the tag `before-cleanup` takes
you back to it (`git checkout before-cleanup`).
