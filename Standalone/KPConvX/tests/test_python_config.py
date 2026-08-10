import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import init_cfg, load_cfg, save_cfg  # noqa: E402
from utils.python_config import (  # noqa: E402
    PythonConfigError,
    apply_config_options,
    apply_python_config,
    parse_config_options,
    resolve_python_config,
)


class PythonConfigTest(unittest.TestCase):

    def write_config(self, directory, name, source):
        path = Path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        return path

    def test_arbitrary_name_inherits_base_and_overrides_selected_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                '任意实验名.py',
                """\
model = dict(fa_enabled=True)
train = dict(amp_dtype='float16')
""",
            )
            base_cfg = init_cfg()
            base_cfg.exp.seed = 123

            updated_cfg, resolved_path = apply_python_config(base_cfg, config_path)

            self.assertTrue(updated_cfg.model.fa_enabled)
            self.assertEqual(updated_cfg.train.amp_dtype, 'float16')
            self.assertEqual(updated_cfg.exp.seed, 123)
            self.assertFalse(base_cfg.model.fa_enabled)
            self.assertEqual(base_cfg.train.amp_dtype, 'bfloat16')
            self.assertEqual(resolved_path, os.path.realpath(config_path))
            self.assertEqual(updated_cfg.exp.config_file, os.path.realpath(config_path))
            self.assertEqual(len(updated_cfg.exp.config_sources), 1)
            expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.assertEqual(
                updated_cfg.exp.config_sources[0]['sha256'], expected_hash
            )

    def test_resolves_name_without_suffix_relative_to_config_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                's3dis/demo.py',
                "train = dict(amp_enabled=True)\n",
            )

            resolved_path = resolve_python_config(
                's3dis/demo', config_root=temp_dir
            )

            self.assertEqual(resolved_path, os.path.realpath(config_path))

    def test_recursive_base_inheritance_and_delete_replace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_config(
                temp_dir,
                '_base_/adapter.py',
                """\
model = dict(fa_enabled=True, fa_num_anchors=48)
train = dict(lr_decays={'10': 0.5})
""",
            )
            config_path = self.write_config(
                temp_dir,
                's3dis/child.py',
                """\
_base_ = '../_base_/adapter.py'
model = dict(kp_mode='kpconvd')
train = dict(lr_decays=dict(_delete_=True, **{'100': 0.1}))
""",
            )

            updated_cfg, _ = apply_python_config(init_cfg(), config_path)

            self.assertTrue(updated_cfg.model.fa_enabled)
            self.assertEqual(updated_cfg.model.fa_num_anchors, 48)
            self.assertEqual(updated_cfg.model.kp_mode, 'kpconvd')
            self.assertEqual(dict(updated_cfg.train.lr_decays), {'100': 0.1})
            self.assertEqual(len(updated_cfg.exp.config_sources), 2)

    def test_sibling_base_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_config(
                temp_dir,
                'first.py',
                "model = dict(kp_mode='kpconvd')\n",
            )
            self.write_config(
                temp_dir,
                'second.py',
                "model = dict(kp_mode='kpconvx')\n",
            )
            child_path = self.write_config(
                temp_dir,
                'child.py',
                "_base_ = ['./first.py', './second.py']\n",
            )

            with self.assertRaisesRegex(PythonConfigError, 'model.kp_mode'):
                apply_python_config(init_cfg(), child_path)

    def test_sibling_bases_may_override_different_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.write_config(
                temp_dir,
                'first.py',
                "model = dict(kp_mode='kpconvd')\n",
            )
            self.write_config(
                temp_dir,
                'second.py',
                "train = dict(amp_enabled=True)\n",
            )
            child_path = self.write_config(
                temp_dir,
                'child.py',
                "_base_ = ['./first.py', './second.py']\n",
            )

            updated_cfg, _ = apply_python_config(init_cfg(), child_path)

            self.assertEqual(updated_cfg.model.kp_mode, 'kpconvd')
            self.assertTrue(updated_cfg.train.amp_enabled)

    def test_unknown_parameter_is_rejected_without_mutating_base(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                'typo.py',
                "train = dict(amp_enable=True)\n",
            )
            base_cfg = init_cfg()

            with self.assertRaisesRegex(PythonConfigError, 'train.amp_enable'):
                apply_python_config(base_cfg, config_path)

            self.assertNotIn('amp_enable', base_cfg.train)
            self.assertFalse(base_cfg.train.amp_enabled)

    def test_unknown_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                'unknown_section.py',
                "optimizer = dict(type='AdamW')\n",
            )

            with self.assertRaisesRegex(PythonConfigError, 'unknown section'):
                apply_python_config(init_cfg(), config_path)

    def test_exception_is_wrapped_without_mutating_base(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                'broken.py',
                "raise RuntimeError('intentional failure')\n",
            )
            base_cfg = init_cfg()

            with self.assertRaisesRegex(PythonConfigError, 'intentional failure'):
                apply_python_config(base_cfg, config_path)

            self.assertFalse(base_cfg.train.amp_enabled)

    def test_circular_inheritance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = self.write_config(
                temp_dir,
                'first.py',
                "_base_ = './second.py'\n",
            )
            self.write_config(
                temp_dir,
                'second.py',
                "_base_ = './first.py'\n",
            )

            with self.assertRaisesRegex(PythonConfigError, 'Circular'):
                apply_python_config(init_cfg(), first_path)

    def test_complete_section_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                'replace_section.py',
                "model = dict(_delete_=True, kp_mode='kpconvd')\n",
            )

            with self.assertRaisesRegex(PythonConfigError, 'complete model section'):
                apply_python_config(init_cfg(), config_path)

    def test_generic_options_parse_literals_and_preserve_base(self):
        base_cfg = init_cfg()
        options = [
            'train.amp_enabled=true',
            'train.amp_dtype=float16',
            'model.layer_blocks=(2, 2, 2, 6, 2)',
            'model.neighbor_limits=[12, 16, 20]',
        ]

        updated_cfg = apply_config_options(base_cfg, options)

        self.assertTrue(updated_cfg.train.amp_enabled)
        self.assertEqual(updated_cfg.train.amp_dtype, 'float16')
        self.assertEqual(updated_cfg.model.layer_blocks, (2, 2, 2, 6, 2))
        self.assertEqual(updated_cfg.model.neighbor_limits, [12, 16, 20])
        self.assertEqual(updated_cfg.exp.config_options, options)
        self.assertFalse(base_cfg.train.amp_enabled)
        self.assertEqual(base_cfg.exp.config_options, [])

    def test_generic_options_reject_unknown_duplicate_and_malformed_keys(self):
        with self.assertRaisesRegex(PythonConfigError, 'unknown parameter'):
            apply_config_options(init_cfg(), ['train.amp_enable=true'])
        with self.assertRaisesRegex(PythonConfigError, 'more than once'):
            parse_config_options(
                ['train.amp_enabled=true', 'train.amp_enabled=false']
            )
        with self.assertRaisesRegex(PythonConfigError, 'section.key=value'):
            parse_config_options(['train.amp_enabled'])

    def test_provenance_fields_are_managed_automatically(self):
        base_cfg = init_cfg()
        self.assertEqual(base_cfg.exp.config_file, '')
        self.assertEqual(base_cfg.exp.config_sources, [])
        self.assertEqual(base_cfg.exp.config_options, [])

        with self.assertRaisesRegex(PythonConfigError, 'managed automatically'):
            apply_config_options(base_cfg, ['exp.config_file=fake.py'])

    def test_provenance_survives_effective_config_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = self.write_config(
                temp_dir,
                'profile.py',
                "train = dict(amp_enabled=True)\n",
            )
            cfg, _ = apply_python_config(init_cfg(), config_path)
            cfg = apply_config_options(cfg, ['model.fa_num_anchors=80'])
            cfg.exp.log_dir = temp_dir

            save_cfg(cfg, temp_dir)
            restored_cfg = load_cfg(temp_dir)

            self.assertEqual(restored_cfg.exp.config_file, cfg.exp.config_file)
            self.assertEqual(restored_cfg.exp.config_sources, cfg.exp.config_sources)
            self.assertEqual(restored_cfg.exp.config_options, cfg.exp.config_options)


if __name__ == '__main__':
    unittest.main()
