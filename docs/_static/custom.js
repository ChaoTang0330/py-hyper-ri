document.addEventListener("DOMContentLoaded", () => {
  const insertTitle = (target, text) => {
    if (!target || target.previousElementSibling?.classList.contains("hyri-autodoc-section-title")) {
      return;
    }

    const title = document.createElement("p");
    title.className = "rubric hyri-autodoc-section-title";
    title.textContent = text;
    target.parentNode.insertBefore(title, target);
  };

  const firstFieldName = (fieldList) => {
    return fieldList?.querySelector(":scope > dt")?.textContent?.replace(/:$/, "").trim();
  };

  const isNote = (element) => {
    return element?.matches("div.admonition.note, div.admonition-notes");
  };

  document.querySelectorAll("dl.py.class > dd").forEach((classBody) => {
    const directChildren = Array.from(classBody.children);
    const shapeField = directChildren.find((child) => {
      return child.matches("dl.field-list") && firstFieldName(child) === "Shape";
    });
    if (shapeField) {
      shapeField.classList.add("hyri-shape-field");
    }

    const firstMethod = directChildren.find((child) => child.matches("dl.py.method"));
    const initField = directChildren.find((child) => {
      return child.matches("dl.field-list") &&
        firstFieldName(child) === "Parameters" &&
        (!firstMethod || directChildren.indexOf(child) < directChildren.indexOf(firstMethod));
    });
    if (initField) {
      const initSummary = initField.previousElementSibling?.tagName === "P"
        ? initField.previousElementSibling
        : null;
      const initNotes = [];
      let nextInitElement = initField.nextElementSibling;
      while (isNote(nextInitElement)) {
        initNotes.push(nextInitElement);
        nextInitElement = nextInitElement.nextElementSibling;
      }
      const initTitle = document.createElement("p");
      initTitle.className = "rubric hyri-autodoc-section-title";
      initTitle.textContent = "Initialisation";

      const sectionStart = shapeField ??
        classBody.querySelector(":scope > dl.py.attribute, :scope > p.rubric, :scope > dl.py.method");
      const initFragment = document.createDocumentFragment();
      initFragment.appendChild(initTitle);
      if (initSummary) {
        initFragment.appendChild(initSummary);
      }
      initFragment.appendChild(initField);
      initNotes.forEach((note) => initFragment.appendChild(note));
      classBody.insertBefore(initFragment, sectionStart);
    }

    insertTitle(shapeField, "Shape");

    classBody.querySelectorAll(":scope > dl.py.method").forEach((method) => {
      if (method.querySelector(":scope > dt")?.id?.endsWith(".__init__")) {
        method.remove();
      }
    });

    insertTitle(classBody.querySelector(":scope > dl.py.attribute"), "Attributes");
    insertTitle(classBody.querySelector(":scope > dl.py.method"), "Methods");
  });
});
