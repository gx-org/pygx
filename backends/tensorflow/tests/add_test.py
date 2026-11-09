# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test GX with a TensorFlow backend."""

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from google3.third_party.gxlang.pygx.backends.tensorflow import gxtf
from google3.third_party.gxlang.pygx.backends.tensorflow.tests import gxtftests


class TfAddTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gxtf.verbose_error(True)

  def tf_add(self, a, b):
    """Takes two arguments and returns their sum using TensorFlow."""
    tensor_a = tf.convert_to_tensor(a)
    tensor_b = tf.convert_to_tensor(b)
    return gxtftests.Add(tensor_a, tensor_b)

  def test_tf_add_with_scalars(self):
    """Tests addition with simple scalar float values."""
    x = 5.0
    y = 3.5
    expected = 8.5

    result = self.tf_add(x, y)

    self.assertAllClose(result, expected)

  def test_tf_add_with_tensors(self):
    """Tests addition directly using TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    y = tf.constant([3.0, 4.0])
    expected = [4.0, 6.0]

    result = self.tf_add(x, y)

    self.assertAllClose(result, expected)

  def test_tf_add_float32_with_tensors(self):
    """Tests AddFloat32 with TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    y = tf.constant([3.0, 4.0])
    expected = [4.0, 6.0]

    result = gxtftests.AddFloat32(tf.shape(x), x, y)

    self.assertAllClose(result, expected)

  def test_tf_add_one_float32_with_tensors(self):
    """Tests AddOneFloat32 with TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    expected = [2.0, 3.0]

    result = gxtftests.AddOneFloat32(tf.shape(x), x)

    self.assertAllClose(result, expected)

  def test_tf_add_float32_one_with_tensors(self):
    """Tests AddFloat32One with TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    expected = [2.0, 3.0]

    result = gxtftests.AddFloat32One(tf.shape(x), x)

    self.assertAllClose(result, expected)

  def test_tf_add_float32_expr_with_tensors(self):
    """Tests AddFloat32Expr with TensorFlow Tensors."""
    x = tf.constant([1.0, 2.0])
    y = tf.constant([3.0, 4.0])
    expected = [5.0, 7.0]

    result = gxtftests.AddFloat32Expr(tf.shape(x), x, y)

    self.assertAllClose(result, expected)

  def test_tf_div_float32_with_tensors(self):
    """Tests DivFloat32 with TensorFlow Tensors."""
    x = tf.constant([6.0, 8.0])
    y = tf.constant([2.0, 4.0])
    expected = [3.0, 2.0]

    result = gxtftests.DivFloat32(tf.shape(x), x, y)

    self.assertAllClose(result, expected)

  def test_tf_add_in_tf_function(self):
    """Tests Add, AddFloat32, and AddOneFloat32 inside tf.function and XLA."""

    def fn(a, b):
      return (
          gxtftests.Add(a, b),
          gxtftests.AddFloat32(tf.shape(a), a, b),
          gxtftests.AddOneFloat32(tf.shape(a), a),
      )

    x = tf.constant([1.0, 2.0])
    y = tf.constant([3.0, 4.0])
    for compiled_fn in (tf.function(fn), tf.function(fn, jit_compile=True)):
      res_add, res_add_f32, res_add_one = compiled_fn(x, y)
      self.assertAllClose(res_add, [4.0, 6.0])
      self.assertAllClose(res_add_f32, [4.0, 6.0])
      self.assertAllClose(res_add_one, [2.0, 3.0])

  def test_tf_add_gradient(self):
    """Tests gradients through AddFloat32 and AddOneFloat32 in eager, graph, and XLA modes."""

    def grad_fn(a, b):
      with tf.GradientTape() as tape:
        tape.watch([a, b])
        z = gxtftests.AddOneFloat32(
            tf.shape(a), gxtftests.AddFloat32(tf.shape(a), a, b)
        )
      return tape.gradient(z, [a, b])

    x = tf.constant([1.0, 2.0])
    y = tf.constant([3.0, 4.0])
    for fn in (
        grad_fn,
        tf.function(grad_fn),
        tf.function(grad_fn, jit_compile=True),
    ):
      dx, dy = fn(x, y)
      self.assertAllClose(dx, [1.0, 1.0])
      self.assertAllClose(dy, [1.0, 1.0])

  def test_tf_add_len_with_python_int(self):
    """Tests AddLen with a Python int generic parameter."""
    x = tf.constant([1.0, 2.0, 4.0])
    expected = [4.0, 5.0, 7.0]

    result = gxtftests.AddLen(3, x)

    self.assertAllClose(result, expected)

  def test_tf_add_float32_with_python_shapes(self):
    """Tests AddFloat32 with Python list, tuple, and numpy array shapes."""
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    expected = [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]

    for shape in (
        x.shape.as_list(),
        tuple(x.shape.as_list()),
        np.array(x.shape.as_list(), dtype=np.int64),
    ):
      self.assertAllClose(gxtftests.AddFloat32(shape, x, y), expected)

  def test_tf_add_float32_with_scalar_python_shape(self):
    """Tests AddFloat32 with an empty Python list shape for a scalar tensor."""
    x = tf.constant(1.5)
    y = tf.constant(2.5)

    result = gxtftests.AddFloat32(x.shape.as_list(), x, y)

    self.assertAllClose(result, 4.0)


if __name__ == "__main__":
  absltest.main()
