# Project Setup Guide for Fall-2025-Team-Big-Data

> This guide will help you get started contributing to the project — including installing dependencies, setting up your Python environment, and contributing using GitHub.

---

## 0. Prerequisites

Please install the following tools before you start:

| Tool | Purpose | Link |
|------|---------|------|
| **Anaconda** | Manage Python + dependencies in an isolated environment | https://www.anaconda.com/download |
| **GitHub Desktop** | Simple graphical app for Git repo management | https://desktop.github.com |
| **Visual Studio Code (VS Code)** | Code editor with Git and Python integration | https://code.visualstudio.com |

All are free and available for macOS and Windows.

---

## 1. Clone the GitHub Repository

> This creates a local copy of the project linked to the GitHub remote.

1. Open GitHub Desktop.
2. Go to `File ▷ Clone Repository...`
3. Use this URL:  
   ```
   https://github.com/arvindsuresh-math/Fall-2025-Team-Big-Data.git
   ```
4. Choose a local folder and click **Clone**.
5. Open the repo in VS Code via **Repository ▷ Open in Visual Studio Code**.

---

## 2. Set Up the Conda Environment (First-Time Only)

1. Open a terminal (macOS: Terminal app, Windows: Anaconda Prompt).
2. Navigate to your local repo folder.
3. Run the following command to create the environment:

    ```bash
    conda env create -f environment.yml
    ```

4. Activate the environment:

    ```bash
    conda activate airbnb-project
    ```

5. (Optional) To use the environment in Jupyter:

    ```bash
    python -m ipykernel install --user --name=airbnb-project
    ```

6. In JupyterLab or VS Code notebooks, select the `airbnb-project` kernel.

---

## 3. Securely Manage Secrets

If you need to store credentials (e.g., AWS, API keys), do the following:

1. Create a `.env` file in the project root.
2. Add secrets like this:

    ```env
    AWS_ACCESS_KEY_ID=your_key_here
    AWS_SECRET_ACCESS_KEY=your_secret_here
    ```

3. Do **not** commit this file. It is ignored via `.gitignore`.

4. To load these in Python:

    ```python
    from dotenv import load_dotenv
    load_dotenv()
    ```

Install `python-dotenv` if needed:

```bash
pip install python-dotenv
```

---

## 4. Create Your Own Branch (One-Time Setup)

> Never work directly on the `main` branch.

1. In GitHub Desktop:
    - Click **Current Branch ▷ New Branch**
    - Name it something like `yourname-dev`
    - Base branch = `main`
    - Click **Create Branch** then **Publish Branch**

2. You now have a personal draft area for your work.

---

## 5. Daily Workflow

Here’s the workflow you should follow each time you sit down to work:

1. **Open the repo in VS Code**
2. **Open a terminal and activate the environment**:
    ```bash
    conda activate airbnb-project
    ```
3. **Make sure you're on your own branch**:
    ```bash
    git branch
    git checkout yourname-dev
    ```
4. **Pull updates from `main` periodically**:
    ```bash
    git pull origin main
    ```
5. **Do your work**: notebooks, scripts, etc.
6. **Commit and push changes**:
    - Use GitHub Desktop or run:
      ```bash
      git add .
      git commit -m "Your commit message"
      git push origin yourname-dev
      ```

---

## 6. First-Time .env Setup Example

Create a file named `.env` in the project root:

```env
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

Do **not** share this file or commit it to Git.

---

## 7. Summary Checklist ✅

- [ ] Install Anaconda, GitHub Desktop, and VS Code
- [ ] Clone the repo using GitHub Desktop
- [ ] Create and activate the Conda environment
- [ ] Create a `.env` file for secrets (but don’t commit it!)
- [ ] Create your own feature branch
- [ ] Commit + push your changes from your branch
- [ ] Pull from `main` regularly to stay up to date

---
