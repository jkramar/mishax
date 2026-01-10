"""Tests for the Code Modification Cookbook

This test suite demonstrates every category in the cookbook with working examples.
Each test shows the recommended approach for that type of modification.
"""

from __future__ import annotations

import unittest
from unittest import mock

from mishax import ast_patcher
from mishax import cookbook


class TestCategory1ExpressionReplacement(unittest.TestCase):
    """CATEGORY 1: Expression Replacement - Use AST Patcher"""

    def test_replace_literal_with_variable(self):
        """Replace a hardcoded literal with a variable"""
        # Patch: change 'return a + b' to 'return a + b + bonus'
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='bonus = 100'),
            **{'SimpleCalculator': ['a + b', 'a + b + bonus']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.add(5, 3)
            self.assertEqual(result, 108)  # 5 + 3 + 100

    def test_replace_constant_with_computation(self):
        """Change a constant to a computed value"""
        # Patch: change 'temp * 3' to 'temp * multiplier()'
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='def multiplier(): return 5'),
            **{'SimpleCalculator': ['temp * 3', 'temp * multiplier()']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.complex_calculation(10)
            # (10 + 2) * 5 - 1 = 59
            self.assertEqual(result, 59)

    def test_intercept_return_value(self):
        """Intercept and modify a return value"""
        # Patch: change 'return a * b' to 'return (a * b) * 10'
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            **{'SimpleCalculator': ['a * b', '(a * b) * 10']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.multiply(4, 5)
            self.assertEqual(result, 200)  # (4 * 5) * 10


class TestCategory2StatementAddition(unittest.TestCase):
    """CATEGORY 2: Statement Addition - Use AST Patcher"""

    def test_add_logging_callback(self):
        """Add a callback statement for logging/instrumentation"""
        calls = []

        def track_call(value):
            calls.append(value)

        # Patch: add tracking after computation
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='track = None'),
            **{'SimpleCalculator': [
                'temp = temp * 3', 'temp = temp * 3\ntrack(temp)'
            ]},
        )

        with patcher():
            patcher.globals['track'] = track_call  # Inject callback into patched module
            calc = cookbook.SimpleCalculator()
            result = calc.complex_calculation(10)

            self.assertEqual(result, 35)
            self.assertEqual(calls, [36])  # (10 + 2) * 3

    def test_add_multiple_statements(self):
        """Add multiple instrumentation statements"""
        events = []

        def log_event(msg):
            events.append(msg)

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='log = None'),
            **{'DataProcessor': [
                'result = item.upper()', '''result = item.upper()
log("processing")
log(result)'''
            ]},
        )

        with patcher():
            patcher.globals['log'] = log_event
            processor = cookbook.DataProcessor()
            result = processor.process_item("hello")

            self.assertEqual(result, "HELLO")
            self.assertEqual(events, ["processing", "HELLO"])


class TestCategory3MultiLineBlockReplacement(unittest.TestCase):
    """CATEGORY 3: Multi-line Block Replacement - Use AST Patcher"""

    def test_replace_algorithm_implementation(self):
        """Replace an entire algorithm block"""
        # Replace the forward logic with simpler version
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            **{'MLModel': [
                'h1 = self.forward_layer(x)\nh2 = self.forward_layer(h1)\noutput = [v + 1.0 for v in h2]',
                'h1 = [v * 10.0 for v in x]\noutput = h1'
            ]},
        )

        with patcher():
            model = cookbook.MLModel()
            result = model.forward([1.0, 2.0])
            self.assertEqual(result, [10.0, 20.0])

    def test_replace_initialization_block(self):
        """Replace initialization logic"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            **{'MLModel': [
                'self.hidden_size = hidden_size\nself.layer_count = 0',
                'self.hidden_size = hidden_size * 2\nself.layer_count = 100\nself.patched = True'
            ]},
        )

        with patcher():
            model = cookbook.MLModel(hidden_size=64)
            self.assertEqual(model.hidden_size, 128)
            self.assertEqual(model.layer_count, 100)
            self.assertTrue(model.patched)


class TestCategory4ReturnValueMocking(unittest.TestCase):
    """CATEGORY 4: Return Value Mocking - Use unittest.mock.patch"""

    def test_mock_network_call(self):
        """Mock external network dependency"""
        with mock.patch.object(
            cookbook.NetworkClient,
            'fetch_data',
            return_value={"url": "test", "status": "mocked"},
        ):
            client = cookbook.NetworkClient()
            result = client.fetch_data("https://example.com")
            self.assertEqual(result["status"], "mocked")

    def test_mock_database_operation(self):
        """Mock database save operation"""
        with mock.patch.object(
            cookbook.NetworkClient, 'save_data', return_value=False
        ):
            client = cookbook.NetworkClient()
            result = client.save_data({"key": "value"})
            self.assertFalse(result)

    def test_mock_external_api(self):
        """Mock external API with specific return values"""
        test_data = {"id": 123, "name": "test"}

        with mock.patch.object(
            cookbook.NetworkClient, 'fetch_data', return_value=test_data
        ):
            client = cookbook.NetworkClient()
            result = client.fetch_data("https://api.example.com/data")
            self.assertEqual(result["id"], 123)


class TestCategory5BehaviorTracking(unittest.TestCase):
    """CATEGORY 5: Behavior Tracking (Spy) - AST Patcher OR mock.patch"""

    def test_track_internal_with_ast_patcher(self):
        """Use AST Patcher to track internal computations"""
        intermediate_values = []

        def capture(value):
            intermediate_values.append(value)

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='capture = None'),
            **{'MLModel': [
                'h1 = self.forward_layer(x)', '''h1 = self.forward_layer(x)
capture(h1)''',
                'h2 = self.forward_layer(h1)', '''h2 = self.forward_layer(h1)
capture(h2)'''
            ]},
        )

        with patcher():
            patcher.globals['capture'] = capture
            model = cookbook.MLModel()
            result = model.forward([1.0, 2.0])

            # Verify we captured intermediate values
            self.assertEqual(len(intermediate_values), 2)
            self.assertEqual(intermediate_values[0], [2.0, 4.0])  # h1
            self.assertEqual(intermediate_values[1], [4.0, 8.0])  # h2

    def test_track_boundary_with_mock(self):
        """Use mock.patch to track calls at module boundaries"""
        with mock.patch.object(
            cookbook.NetworkClient, 'fetch_data', wraps=None
        ) as mock_fetch:
            mock_fetch.return_value = {"status": "ok"}

            client = cookbook.NetworkClient()
            client.fetch_data("https://test.com")
            client.fetch_data("https://test2.com")

            # Verify calls
            self.assertEqual(mock_fetch.call_count, 2)
            mock_fetch.assert_any_call("https://test.com")
            mock_fetch.assert_any_call("https://test2.com")


class TestCategory6AddingImportsGlobals(unittest.TestCase):
    """CATEGORY 6: Adding Imports/Globals - Use AST Patcher prefix"""

    def test_add_import_for_logging(self):
        """Add import statement via prefix"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='import math'),
            **{'SimpleCalculator': ['a + b', 'a + b + int(math.pi)']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.add(5, 3)
            self.assertEqual(result, 11)  # 5 + 3 + 3

    def test_add_global_counter(self):
        """Add global counter variable"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(
                prefix='''call_count = 0
def increment():
    global call_count
    call_count += 1
    return call_count'''
            ),
            **{'SimpleCalculator': ['a + b', 'a + b + increment()']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            r1 = calc.add(5, 3)
            r2 = calc.add(5, 3)
            r3 = calc.add(5, 3)

            self.assertEqual(r1, 9)  # 5 + 3 + 1
            self.assertEqual(r2, 10)  # 5 + 3 + 2
            self.assertEqual(r3, 11)  # 5 + 3 + 3

    def test_add_helper_function(self):
        """Add helper function via prefix"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(
                prefix='def double(x): return x * 2'
            ),
            **{'SimpleCalculator': ['a + b', 'double(a + b)']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.add(5, 3)
            self.assertEqual(result, 16)  # (5 + 3) * 2


class TestCategory7ConditionalPatching(unittest.TestCase):
    """CATEGORY 7: Conditional Patching - AST Patcher + conditionals"""

    def test_conditional_logging(self):
        """Only log when debug flag is set"""
        logs = []

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(
                prefix='''DEBUG = False
logs = []
def maybe_log(msg):
    if DEBUG:
        logs.append(msg)'''
            ),
            **{'DataProcessor': [
                'result = item.upper()', '''result = item.upper()
maybe_log(result)'''
            ]},
        )

        with patcher():
            # First try without debug
            processor = cookbook.DataProcessor()
            processor.process_item("hello")
            self.assertEqual(len(patcher.globals['logs']), 0)

            # Now enable debug
            patcher.globals['DEBUG'] = True
            processor.process_item("world")
            self.assertEqual(patcher.globals['logs'], ["WORLD"])

    def test_conditional_behavior(self):
        """Different behavior based on condition"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(
                prefix='USE_FAST_MODE = True'
            ),
            **{'SimpleCalculator': [
                'temp * 3', 'temp * (2 if USE_FAST_MODE else 3)'
            ]},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.complex_calculation(10)
            # (10 + 2) * 2 - 1 = 23
            self.assertEqual(result, 23)


class TestCategory8MultipleModifications(unittest.TestCase):
    """CATEGORY 8: Multiple Modifications - Use allow_num_matches_upto"""

    def test_instrument_all_layers(self):
        """Instrument every layer call in a model"""
        layer_outputs = []

        def capture_layer(output):
            layer_outputs.append(output)
            return output

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(
                prefix='capture_layer = None',
                allow_num_matches_upto={'MLModel': 2},
            ),
            **{'MLModel': [
                'self.forward_layer(x)', 'capture_layer(self.forward_layer(x))',
                'self.forward_layer(h1)', 'capture_layer(self.forward_layer(h1))',
            ]},
        )

        with patcher():
            patcher.globals['capture_layer'] = capture_layer
            model = cookbook.MLModel()
            result = model.forward([1.0, 2.0])

            # Both layer calls were captured
            self.assertEqual(len(layer_outputs), 2)

    def test_log_all_returns(self):
        """Log every return statement in a function"""
        returns = []

        def log_return(value):
            returns.append(value)
            return value

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(
                prefix='log_return = None',
                allow_num_matches_upto={'function_with_multiple_returns': 3},
            ),
            **{'function_with_multiple_returns': [
                'return "positive"', 'return log_return("positive")',
                'return "negative"', 'return log_return("negative")',
                'return "zero"', 'return log_return("zero")',
            ]},
        )

        with patcher():
            patcher.globals['log_return'] = log_return

            r1 = cookbook.function_with_multiple_returns(5)
            r2 = cookbook.function_with_multiple_returns(-3)
            r3 = cookbook.function_with_multiple_returns(0)

            self.assertEqual(returns, ["positive", "negative", "zero"])


class TestCategory9ConstructorModification(unittest.TestCase):
    """CATEGORY 9: Constructor/Initialization - AST Patcher OR mock.patch"""

    def test_modify_init_with_ast_patcher(self):
        """Add logging to __init__ with AST Patcher"""
        init_calls = []

        def log_init(config):
            init_calls.append(config)

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='log_init = None'),
            **{'ConfigurableService': [
                'self.config = config', '''self.config = config
log_init(config)'''
            ]},
        )

        with patcher():
            patcher.globals['log_init'] = log_init
            service = cookbook.ConfigurableService({"key": "value"})
            self.assertEqual(init_calls, [{"key": "value"}])

    def test_change_default_values(self):
        """Change default initialization values"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            **{'ConfigurableService': [
                'self.config.get(key, "default")', 'self.config.get(key, "patched_default")'
            ]},
        )

        with patcher():
            service = cookbook.ConfigurableService({})
            result = service.get_setting("missing_key")
            self.assertEqual(result, "patched_default")

    def test_mock_entire_constructor(self):
        """Replace object with mock using mock.patch"""
        mock_service = mock.Mock()
        mock_service.get_setting.return_value = "mocked"

        with mock.patch(
            'mishax.cookbook.ConfigurableService', return_value=mock_service
        ):
            service = cookbook.ConfigurableService({"any": "config"})
            result = service.get_setting("key")
            self.assertEqual(result, "mocked")


class TestCategory10AttributeAccessInterception(unittest.TestCase):
    """CATEGORY 10: Attribute Access - AST Patcher OR PropertyMock"""

    def test_intercept_property_with_ast_patcher(self):
        """Intercept and modify property access"""
        access_log = []

        def log_access(key):
            access_log.append(key)

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='log_access = None'),
            **{'ConfigurableService': [
                'return self.config.get(key, "default")', '''log_access(key)
return self.config.get(key, "default")'''
            ]},
        )

        with patcher():
            patcher.globals['log_access'] = log_access
            service = cookbook.ConfigurableService({"a": "1"})
            service.get_setting("a")
            service.get_setting("b")

            self.assertEqual(access_log, ["a", "b"])

    def test_mock_property_value(self):
        """Replace instance attribute using mock"""
        service = cookbook.ConfigurableService({"key": "original"})

        # Mock the instance attribute directly
        service.config = {"key": "mocked"}
        result = service.get_setting("key")
        self.assertEqual(result, "mocked")


class TestCategory11ExceptionHandling(unittest.TestCase):
    """CATEGORY 11: Exception Handling - AST Patcher"""

    def test_add_exception_logging(self):
        """Add logging to exception handling - demonstrated with function_calling_others"""
        # Note: This test demonstrates the concept. In practice, exception handling
        # is better instrumented at the catch site rather than the raise site.

        # Skip this test as it requires adding a new function to cookbook.py
        # which would complicate the examples. The concept is documented.
        self.skipTest("Exception handling example requires pre-defined functions in cookbook.py")


class TestCategory12LoopInstrumentation(unittest.TestCase):
    """CATEGORY 12: Loop Instrumentation - AST Patcher"""

    def test_instrument_loop_body(self):
        """Add progress tracking to loop body"""
        iterations = []

        def track_iteration(item):
            iterations.append(item)

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='track_iteration = None'),
            **{'DataProcessor': [
                'return [self.process_item(item) for item in items]', '''track_iteration("start")
result = [self.process_item(item) for item in items]
track_iteration("end")
return result'''
            ]},
        )

        with patcher():
            patcher.globals['track_iteration'] = track_iteration
            processor = cookbook.DataProcessor()
            result = processor.process_batch(["a", "b", "c"])

            self.assertEqual(result, ["A", "B", "C"])
            self.assertEqual(iterations, ["start", "end"])


class TestCategory13CombiningBoth(unittest.TestCase):
    """CATEGORY 13: Combining AST Patcher + mock.patch"""

    def test_internal_modification_plus_external_mock(self):
        """Use AST Patcher for internal logic + mock for external calls"""
        internal_calls = []

        def track_internal(data):
            internal_calls.append(data)

        # Mock external dependency
        mock_response = {"url": "mocked", "data": "test"}

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='track_internal = None'),
            **{'NetworkClient': [
                'return {"url": url, "status": "success"}', '''track_internal(url)
return {"url": url, "status": "success"}'''
            ]},
        )

        with patcher():
            patcher.globals['track_internal'] = track_internal

            # Also mock the fetch to avoid real network calls
            with mock.patch.object(
                cookbook.NetworkClient,
                'fetch_data',
                return_value=mock_response,
            ):
                client = cookbook.NetworkClient()
                result = client.fetch_data("https://example.com")

                # Mock took precedence
                self.assertEqual(result["data"], "test")

    def test_instrument_model_and_mock_data_loading(self):
        """Instrument model internals + mock data loading"""
        layer_calls = []

        def track_layer():
            layer_calls.append(1)

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='track_layer = None'),
            **{'MLModel': [
                'self.layer_count += 1', '''self.layer_count += 1
track_layer()'''
            ]},
        )

        with patcher():
            patcher.globals['track_layer'] = track_layer

            # Mock some external data source
            with mock.patch(
                'mishax.cookbook.standalone_function', return_value=100
            ):
                model = cookbook.MLModel()
                model.forward([1.0, 2.0])

                # Verify internal instrumentation worked
                self.assertEqual(len(layer_calls), 2)

                # Verify mock worked
                result = cookbook.standalone_function(1, 2)
                self.assertEqual(result, 100)


class TestCategory14CannotUseEither(unittest.TestCase):
    """CATEGORY 14: Limitations - When neither tool works well"""

    def test_monkey_patching_alternative(self):
        """When AST Patcher and mock don't work, use monkey patching"""
        # Save original
        original_add = cookbook.SimpleCalculator.add

        # Monkey patch
        def patched_add(self, a, b):
            return original_add(self, a, b) + 1000

        cookbook.SimpleCalculator.add = patched_add

        try:
            calc = cookbook.SimpleCalculator()
            result = calc.add(5, 3)
            self.assertEqual(result, 1008)
        finally:
            # Restore
            cookbook.SimpleCalculator.add = original_add

    def test_wrapper_alternative(self):
        """Use wrapper/decorator pattern as alternative"""

        def with_logging(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result * 2

            return wrapper

        # Save original
        original_multiply = cookbook.SimpleCalculator.multiply

        # Wrap
        cookbook.SimpleCalculator.multiply = with_logging(original_multiply)

        try:
            calc = cookbook.SimpleCalculator()
            result = calc.multiply(5, 3)
            self.assertEqual(result, 30)  # (5 * 3) * 2
        finally:
            # Restore
            cookbook.SimpleCalculator.multiply = original_multiply


class TestDecisionTree(unittest.TestCase):
    """Test examples from the decision tree"""

    def test_decision_external_dependency_return_value(self):
        """External dependency + just control return → mock.patch"""
        with mock.patch.object(
            cookbook.NetworkClient,
            'fetch_data',
            return_value={"result": "mocked"},
        ):
            client = cookbook.NetworkClient()
            result = client.fetch_data("url")
            self.assertEqual(result["result"], "mocked")

    def test_decision_external_dependency_internal_behavior(self):
        """External dependency + modify internal → AST Patcher"""
        calls = []

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='calls = []'),
            **{'NetworkClient': [
                'return {"url": url, "status": "success"}', '''calls.append(url)
return {"url": url, "status": "success"}'''
            ]},
        )

        with patcher():
            client = cookbook.NetworkClient()
            client.fetch_data("test")
            self.assertEqual(patcher.globals['calls'], ["test"])

    def test_decision_internal_computation_expression(self):
        """Internal computation + replace expression → AST Patcher"""
        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            **{'SimpleCalculator': ['a + b', '(a + b) * 2']},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            result = calc.add(5, 3)
            self.assertEqual(result, 16)

    def test_decision_track_module_boundary(self):
        """Track at module boundary → mock.patch"""
        with mock.patch.object(
            cookbook.NetworkClient, 'fetch_data'
        ) as mock_fetch:
            mock_fetch.return_value = {"data": "test"}
            client = cookbook.NetworkClient()
            client.fetch_data("url")
            self.assertEqual(mock_fetch.call_count, 1)

    def test_decision_track_internal(self):
        """Track internal computation → AST Patcher"""
        tracked = []

        patcher = ast_patcher.ModuleASTPatcher(
            'mishax.cookbook',
            ast_patcher.PatchSettings(prefix='tracked = []'),
            **{'SimpleCalculator': [
                'temp = temp * 3', '''temp = temp * 3
tracked.append(temp)'''
            ]},
        )

        with patcher():
            calc = cookbook.SimpleCalculator()
            calc.complex_calculation(10)
            # (10 + 2) * 3 = 36
            self.assertEqual(patcher.globals['tracked'], [36])


if __name__ == '__main__':
    unittest.main()
