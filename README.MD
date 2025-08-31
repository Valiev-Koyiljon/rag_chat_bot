---

### 1️⃣ Confirm File Structure

Your README expects:

```
mychatbot/
├─ README.md
├─ img/
│  ├─ llm_rag.jpg
│  └─ llm_rag1.jpg
```

Make sure the images are **exactly** in `img/` folder at the same level as README.md.

---

### 2️⃣ Correct Markdown Syntax for JPG

Markdown works fine with JPGs. Your current syntax is okay:

![LLM RAG Screenshot](./img/llm_rag.jpg)
![LLM RAG Screenshot 2](./img/llm_rag1.jpg)


---

### 3️⃣ GitHub Rendering Notes

* GitHub is **case-sensitive**. Make sure the file names match exactly: `llm_rag.jpg` and `llm_rag1.jpg`
* Avoid spaces in filenames. Use underscores `_` or hyphens `-`.
* Ensure you **added the `img` folder and images to Git**:

```bash
git add img/llm_rag.jpg img/llm_rag1.jpg
git commit -m "Add screenshots"
git push
```

---

### 4️⃣ Optional: Use Absolute Path in GitHub

If you want to be sure it works in GitHub, you can use:

```markdown
![LLM RAG Screenshot](./img/llm_rag.jpg)
![LLM RAG Screenshot 2](./img/llm_rag1.jpg)
```

The `./` ensures it looks relative to the README location.

---

✅ After these steps, the images should render in README.md correctly.

---


