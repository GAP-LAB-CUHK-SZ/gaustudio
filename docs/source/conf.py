import pytorch_sphinx_theme

__version__ = "0.0.1"

# -- Project information

project = "gaustudio"
copyright = "2024, Chongjie Ye"
author = "Chongjie Ye, Yuantao Chen"

release = __version__

# -- General configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

# html_theme = "furo"

html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_static_path = ["_static"]
html_css_files = ["css/readthedocs.css"]

# Ignore >>> when copying code
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # The target url that the logo directs to. Unset to do nothing
    "logo_url": "https://github.com/GAP-LAB-CUHK-SZ/gaustudio",
    # "menu" is a list of dictionaries where you can specify the content and the
    # behavior of each item in the menu. Each item can either be a link or a
    # dropdown menu containing a list of links.
    "menu": [
        # A link
        {
            "name": "GitHub",
            "url": "https://github.com/GAP-LAB-CUHK-SZ/gaustudio",
        },
        # A dropdown menu
        # {
        #     "name": "Projects",
        #     "children": [
        #         # A vanilla dropdown item
        #         {
        #             "name": "nerfstudio",
        #             "url": "https://docs.nerf.studio/",
        #             "description": "The all-in-one repo for NeRFs",
        #         },
        #     ],
        #     # Optional, determining whether this dropdown menu will always be
        #     # highlighted.
        #     # "active": True,
        # },
    ],
}
# html_theme_options = {
#     "canonical_url": "",
#     "analytics_id": "",
#     "logo_only": False,
#     "display_version": True,
#     "prev_next_buttons_location": "bottom",
#     "style_external_links": False,
#     # Toc options
#     "collapse_navigation": True,
#     "sticky_navigation": True,
#     "navigation_depth": 4,
#     "includehidden": True,
#     "titles_only": False
# }

# -- Options for EPUB output
epub_show_urls = "footnote"

# typehints
autodoc_typehints = "description"
