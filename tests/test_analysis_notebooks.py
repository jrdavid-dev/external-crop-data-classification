import json
from pathlib import Path
import unittest

REPOSITORY = Path(__file__).resolve().parents[1]


class ActiveNotebookSyntaxTests(unittest.TestCase):
    def test_active_notebook_code_compiles(self):
        for path in (REPOSITORY / 'notebooks').rglob('*.ipynb'):
            with self.subTest(notebook=str(path)):
                book = json.loads(path.read_text())
                for index, cell in enumerate(book['cells']):
                    if cell['cell_type'] == 'code':
                        compile(''.join(cell['source']), f'{path}:cell{index}', 'exec')
