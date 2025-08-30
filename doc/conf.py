from subprocess import check_output

import sphinx

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # support for NumPy-style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinxcontrib.bibtex',
    'sphinx.ext.extlinks',
    'matplotlib.sphinxext.plot_directive',
    'nbsphinx',
]

bibtex_bibfiles = ['references.bib']

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_thumbnails = {
    'example-python-scripts': '_static/thumbnails/soundfigure_level.png',
    'examples/animations-pulsating-sphere': '_static/thumbnails/pulsating_sphere.gif',
}

# Tell autodoc that the documentation is being generated
sphinx.SFS_DOCS_ARE_BEING_BUILT = True

autoclass_content = 'init'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}

autosummary_generate = ['api']

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
}

extlinks = {'sfs': ('https://sfs.readthedocs.io/en/3.2/%s',
                    'https://sfs.rtfd.io/%s')}

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_pre_code = ''
plot_rcparams = {
    'savefig.bbox': 'tight',
}
plot_formats = ['svg', 'pdf']

# use mathjax2 with
# https://github.com/spatialaudio/nbsphinx/issues/572#issuecomment-853389268
# and 'TeX' dictionary
# in future we might switch to mathjax3 once the
# 'begingroup' extension is available
# http://docs.mathjax.org/en/latest/input/tex/extensions/begingroup.html#begingroup
# https://mathjax.github.io/MathJax-demos-web/convert-configuration/convert-configuration.html
mathjax_path = ('https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js'
                '?config=TeX-AMS-MML_HTMLorMML')
mathjax2_config = {
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'processEscapes': True,
        'ignoreClass': 'document',
        'processClass': 'math|output_area',
    },
    'TeX': {
        'extensions': ['newcommand.js', 'begingroup.js'],  # Support for \gdef
    },
}

templates_path = ['_template']

authors = 'SFS Toolbox Developers'
project = 'SFS Toolbox'
copyright = '2019, ' + authors

try:
    release = check_output(['git', 'describe', '--tags', '--always'])
    release = release.decode().strip()
except Exception:
    release = '<unknown>'

try:
    today = check_output(['git', 'show', '-s', '--format=%ad', '--date=short'])
    today = today.decode().strip()
except Exception:
    today = '<unknown date>'

exclude_patterns = ['_build', '**/.ipynb_checkpoints']

default_role = 'any'

jinja_define = r"""
{% set docname = 'doc/' + env.doc2path(env.docname, base=False)|string %}
{% set latex_href = ''.join([
    '\href{https://github.com/sfstoolbox/sfs-python/blob/',
    env.config.release,
    '/',
    docname | escape_latex,
    '}{\sphinxcode{\sphinxupquote{',
    docname | escape_latex,
    '}}}',
]) %}
"""

nbsphinx_prolog = jinja_define + r"""
.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        This page was generated from `{{ docname }}`__.
        Interactive online version:
        :raw-html:`<a href="https://mybinder.org/v2/gh/sfstoolbox/sfs-python/{{ env.config.release }}?filepath={{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

    __ https://github.com/sfstoolbox/sfs-python/blob/
        {{ env.config.release }}/{{ docname }}

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from {{ latex_href }}
    \dotfill}}
"""

nbsphinx_epilog = jinja_define + r"""
.. raw:: latex

    \nbsphinxstopnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{\dotfill\ {{ latex_href }} ends here.}}
"""


# -- Options for HTML output ----------------------------------------------

html_css_files = ['css/title.css']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'navigation_with_keys': True,
}

html_title = project + ", version " + release

html_static_path = ['_static']

html_show_sourcelink = True
html_sourcelink_suffix = ''

htmlhelp_basename = 'SFS'

html_scaled_image_link = False

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'printindex': '',
    'sphinxsetup': r"""
        VerbatimColor={HTML}{F5F5F5},
        VerbatimBorderColor={HTML}{E0E0E0},
        noteBorderColor={HTML}{E0E0E0},
        noteborder=1.5pt,
        warningBorderColor={HTML}{E0E0E0},
        warningborder=1.5pt,
        warningBgColor={HTML}{FBFBFB},
    """,
    'preamble': r"""
\usepackage[sc,osf]{mathpazo}
\linespread{1.05}  % see http://www.tug.dk/FontCatalogue/urwpalladio/
\renewcommand{\sfdefault}{pplj}  % Palatino instead of sans serif
\IfFileExists{zlmtt.sty}{
    \usepackage[light,scaled=1.05]{zlmtt}  % light typewriter font from lmodern
}{
    \renewcommand{\ttdefault}{lmtt}  % typewriter font from lmodern
}
""",
}

latex_documents = [('index', 'SFS.tex', project, authors, 'howto')]

latex_show_urls = 'footnote'

latex_domain_indices = False


# -- Options for epub output ----------------------------------------------

epub_author = authors
