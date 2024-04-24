from keras import    def test_current_path(self):
        """
        Test the current path functionality within name scopes.
        """
        self.assertEqual(current_path(), "outer/middle")
        self.assertEqual(current_path(), "outer")
        self.assertEqual(current_path(), "")

    def test_deduplication(self):
        """
        Test the deduplication of names within name scopes.
        """
        self.assertEqual(current_path(), "")
        with name_scope("name", caller=1):
            with name_scope("name", caller=1):
                self.assertEqual(current_path(), "name")
        self.assertEqual(current_path(), "")
        with name_scope("name"):
            with name_scope("name"):
                self.assertEqual(current_path(), "name/name")

    def test_errors(self):
        """
        Test error handling for invalid inputs to name_scope function.
        """
        with self.assertRaisesRegex(ValueError, "must be a string"):
            name_scope("foo/bar")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            name_scope(4).backend.common.name_scope import current_path
from keras.backend.common.name_scope import name_scope


class NameScopeTest(testing.TestCase):
    def test_stacking(self):
        self.assertEqual(current_path(), "")
        with name_scope("outer") as outer:
            self.assertEqual(outer.name, "outer")
            self.assertEqual(current_path(), "outer")
            with name_scope("middle") as middle:
                self.assertEqual(middle.name, "middle")
                self.assertEqual(current_path(), "outer/middle")
                with name_scope("inner") as inner:
                    self.assertEqual(inner.name, "inner")
                    self.assertEqual(current_path(), "outer/middle/inner")
                self.assertEqual(current_path(), "outer/middle")
            self.assertEqual(current_path(), "outer")
        self.assertEqual(current_path(), "")

    def test_deduplication(self):
        self.assertEqual(current_path(), "")
        with name_scope("name", caller=1):
            with name_scope("name", caller=1):
                self.assertEqual(current_path(), "name")
        self.assertEqual(current_path(), "")
        with name_scope("name"):
            with name_scope("name"):
                self.assertEqual(current_path(), "name/name")

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            name_scope("foo/bar")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            name_scope(4)
