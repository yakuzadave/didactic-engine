# Marimo Notebooks for Didactic Engine

This directory contains [Marimo](https://marimo.io/) notebook versions of the Didactic Engine tutorials. Marimo notebooks are lightweight, reactive Python notebooks that are stored as pure Python files.

## What is Marimo?

Marimo is a modern Python notebook that:
- ✅ **Pure Python**: Notebooks are `.py` files that can be version controlled easily
- ✅ **Reactive**: Cells automatically re-run when dependencies change
- ✅ **Git-friendly**: No JSON bloat, clean diffs, easy merges
- ✅ **Reproducible**: Deterministic execution order
- ✅ **Interactive**: Built-in UI elements and data exploration tools

## Available Notebooks

- **`tutorial_marimo.py`** - Complete tutorial covering installation, usage, and examples of Didactic Engine

## Installation

### Install Marimo

Marimo can be installed as part of the dev extras:

```bash
pip install -e ".[dev]"
```

Or install it separately:

```bash
pip install marimo
```

## Running Marimo Notebooks

### Interactive Mode (Recommended)

To run a notebook in interactive mode with a web interface:

```bash
marimo edit examples/tutorial_marimo.py
```

This will:
1. Start a local web server
2. Open the notebook in your default web browser
3. Allow you to edit and run cells interactively
4. Automatically save changes back to the `.py` file

### Read-only App Mode

To run the notebook as a read-only application:

```bash
marimo run examples/tutorial_marimo.py
```

This is useful for:
- Sharing results with others
- Running demos
- Creating interactive dashboards

### Command Line Execution

You can also run the notebook as a regular Python script:

```bash
python examples/tutorial_marimo.py
```

Note: This will execute all cells in order but won't provide the interactive web interface.

## Converting Jupyter Notebooks to Marimo

If you want to convert other Jupyter notebooks to Marimo format:

```bash
marimo convert your_notebook.ipynb -o your_notebook_marimo.py
```

## Benefits of Marimo Over Jupyter

1. **Version Control**: Marimo notebooks are plain Python files with no JSON metadata
2. **No Hidden State**: Reactive execution prevents out-of-order cell execution bugs
3. **Reproducibility**: Notebooks are guaranteed to run from top to bottom
4. **Collaboration**: Clean Git diffs make code review and collaboration easier
5. **Deployment**: Marimo notebooks can be deployed as web apps with `marimo run`

## Comparison with Jupyter

| Feature | Jupyter (.ipynb) | Marimo (.py) |
|---------|-----------------|--------------|
| File format | JSON | Pure Python |
| Git diffs | Noisy | Clean |
| Execution order | Manual | Reactive |
| Hidden state | Possible | Not possible |
| Deployment | Needs conversion | Native app mode |
| IDE support | Limited | Full Python support |

## Tips for Using Marimo Notebooks

1. **Cell Dependencies**: Marimo tracks dependencies between cells automatically
2. **UI Elements**: Use `mo.ui` for interactive widgets (sliders, dropdowns, etc.)
3. **Markdown**: Use `mo.md()` for markdown cells
4. **Layout**: Use `mo.hstack()`, `mo.vstack()`, and other layout functions
5. **Caching**: Marimo automatically caches expensive computations

## Examples of Marimo Features

### Interactive Widgets

```python
import marimo as mo

# Create a slider
slider = mo.ui.slider(0, 100, value=50, label="Tempo")

# Create a dropdown
option = mo.ui.dropdown(["rock", "jazz", "classical"], value="rock", label="Genre")
```

### Markdown and Layout

```python
mo.md(f"""
# Results
Tempo: {slider.value}
Genre: {option.value}
""")
```

## Troubleshooting

### Port Already in Use

If you get an error that the port is already in use:

```bash
marimo edit examples/tutorial_marimo.py --port 8080
```

### Browser Doesn't Open

Manually navigate to the URL shown in the terminal (usually `http://localhost:2718`).

### Dependencies Not Found

Make sure all dependencies are installed:

```bash
pip install -e ".[all]"  # Install all extras including ML features
```

## Additional Resources

- [Marimo Documentation](https://docs.marimo.io/)
- [Marimo GitHub](https://github.com/marimo-team/marimo)
- [Marimo Discord Community](https://discord.gg/JE7nhX6mD8)
- [Didactic Engine Repository](https://github.com/yakuzadave/didactic-engine)

## Contributing

To add new Marimo notebooks:

1. Create a new notebook: `marimo new examples/my_notebook.py`
2. Or convert an existing Jupyter notebook: `marimo convert notebook.ipynb -o examples/notebook_marimo.py`
3. Test the notebook: `marimo edit examples/my_notebook.py`
4. Commit the `.py` file to the repository

The pure Python format makes it easy to review changes and collaborate!
