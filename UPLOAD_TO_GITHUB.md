# How to put this on GitHub

Written for someone doing this for the first time. Two routes — pick **Route A** if you
just want it online in ten minutes, **Route B** if you want the normal workflow that lets
you update it easily later. You can start with A and move to B whenever.

Delete this file before (or after) uploading — it is instructions for you, not part of the
project.

---

## Before you start

1. **Make a GitHub account** at <https://github.com> if you don't have one. Use your
   university email if you want the free Pro features via
   [GitHub Education](https://education.github.com/pack).
2. **Download and unzip `AdaDiag.zip`.** You should end up with a folder called `AdaDiag`
   containing `README.md`, `src/`, `scripts/`, `tests/`, `results/`, `docs/`.
3. **Check the folder looks right** before uploading:

   ```
   AdaDiag/
   ├── README.md
   ├── LICENSE
   ├── requirements.txt
   ├── environment.yml
   ├── pyproject.toml
   ├── .gitignore
   ├── src/adadiag/      (12 .py files)
   ├── scripts/          (4 .py files)
   ├── tests/            (1 .py file)
   ├── results/          (.json files + README.md)
   └── docs/             (architecture.md)
   ```

---

## Route A — the web browser (no software to install)

**Step 1 — create the empty repository.**

- Go to <https://github.com/new>.
- **Repository name:** `AdaDiag`
- **Description:** `Adaptive modality × task weighting for multimodal microservice failure diagnosis`
- Choose **Public** (so your professor can see it without being added).
- **Do not** tick "Add a README file", "Add .gitignore" or "Choose a license" — the project
  already has all three, and ticking them creates conflicts you'd have to resolve.
- Click **Create repository**.

**Step 2 — upload the files.**

- On the new empty repo page, click the **uploading an existing file** link
  (or go to `https://github.com/<your-username>/AdaDiag/upload/main`).
- Open your `AdaDiag` folder on your computer, select **everything inside it**
  (`Ctrl+A` on Windows / `Cmd+A` on Mac) and **drag it onto the browser window**.
  - Drag the *contents*, not the `AdaDiag` folder itself — otherwise you get
    `AdaDiag/AdaDiag/README.md`.
  - Chrome and Edge preserve subfolders when you drag. Safari sometimes does not — if the
    subfolders don't appear, use Route B instead.
- Wait for every file to finish uploading (you'll see them listed).
- In **Commit changes** at the bottom, type `Initial commit: AdaDiag implementation and results`.
- Click **Commit changes**.

**Step 3 — check it.** Visit `https://github.com/<your-username>/AdaDiag`. The README should
render automatically, and the workflow diagram should appear as a picture (GitHub renders
Mermaid diagrams natively). If the diagram shows as a code block instead, the file was
uploaded as something other than `README.md` — check the name.

Done. Send your professor the URL.

---

## Route B — Git on your computer (the normal way)

Worth doing once: after this, publishing a change is three commands.

**Step 1 — install Git.**

- **Windows:** <https://git-scm.com/download/win>, accept all defaults. Then open
  **Git Bash** from the Start menu.
- **macOS:** open Terminal and type `git --version`; it offers to install if missing.
- **Linux:** `sudo apt install git`

**Step 2 — tell Git who you are** (once per computer):

```bash
git config --global user.name "Shehreyar Rashid"
git config --global user.email "your-github-email@example.com"
```

Use the same email as your GitHub account, or your commits won't be linked to your profile.

**Step 3 — create the empty repository on GitHub** exactly as in Route A Step 1 (public,
nothing ticked).

**Step 4 — upload from your computer.** Open a terminal **inside the `AdaDiag` folder**
(Windows: right-click the folder → *Open Git Bash here*), then:

```bash
git init                      # start tracking this folder
git add .                     # stage every file
git commit -m "Initial commit: AdaDiag implementation and results"
git branch -M main            # name the branch 'main'
git remote add origin https://github.com/<your-username>/AdaDiag.git
git push -u origin main       # upload
```

Replace `<your-username>` with your actual GitHub username.

**Step 5 — the password prompt.** GitHub **does not accept your account password here**.
When it asks:

- **Username:** your GitHub username
- **Password:** a **Personal Access Token**, not your password.

To make one: GitHub → your avatar (top right) → **Settings** → **Developer settings** →
**Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**. Give
it a name, set an expiry, tick the **`repo`** checkbox, generate, and **copy it immediately** —
it is shown only once. Paste it as the password.

(On Windows, Git Credential Manager usually opens a browser window instead and you can just
sign in. Easier — take that if offered.)

**Step 6 — later changes.** Any time you edit something:

```bash
git add .
git commit -m "describe what changed"
git push
```

---

## Things that commonly go wrong

| Symptom | Cause and fix |
|---|---|
| `fatal: not a git repository` | You're not inside the `AdaDiag` folder. `cd` into it first. |
| `Authentication failed` on push | You used your password. Use a Personal Access Token (Step 5). |
| `Updates were rejected... fetch first` | You ticked "Add a README" when creating the repo, so GitHub and your folder both have commits. Fix: `git pull --rebase origin main` then `git push`. |
| `remote origin already exists` | You ran `git remote add` twice. Use `git remote set-url origin <url>`. |
| README shows raw text, diagram as code | The file isn't named exactly `README.md`, or the ```` ```mermaid ```` fence got mangled. |
| `__pycache__` folders appear on GitHub | They were committed before `.gitignore` applied. `git rm -r --cached __pycache__` then commit. |

---

## Two things to check before you share the link

1. **Don't commit the datasets.** The `.gitignore` already excludes `data/` and `TVDiag/`.
   GitHub rejects files over 100 MB, and the README explains how to obtain the data instead —
   which is the right way to do it.
2. **Nothing private.** No API tokens, no passwords, no unpublished material from anyone
   else. Once a public repo exists, assume anything in it has been seen. If you ever do
   commit a secret, deleting the file is **not** enough — the history keeps it; you must
   revoke the secret itself.

---

## Making it look professional (optional, 5 minutes)

On the repo page, click the **gear icon** next to *About* on the right:

- **Description:** `Adaptive modality × task weighting for multimodal microservice failure diagnosis`
- **Topics:** `aiops`, `microservices`, `failure-diagnosis`, `multimodal`, `graph-neural-networks`, `pytorch`, `reproducibility`
- Tick **Releases** off if you have none.

Then **Settings → General → Features** and untick *Wikis* and *Projects* if you won't use
them, so the tab bar stays clean.

If you want a version you can cite in the report, use **Releases → Create a new release**,
tag `v1.0.0`, and GitHub will build a permanent zip of that exact state.
