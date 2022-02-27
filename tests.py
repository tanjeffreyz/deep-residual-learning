import unittest
import torch
import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[~] Using {device}')



class TestDimensions34Layer(unittest.TestCase):
    def test_plain(self):
        batch_size = 128
        model = models.ResNet34(residual=False).to(device)
        test_input = torch.rand(batch_size, 1, 224, 224).to(device)
        result = model.forward(test_input)
        shape = result.shape

        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], batch_size)
        self.assertEqual(shape[1], 1000)

        del model, test_input

    def test_residual_A(self):
        batch_size = 128
        model = models.ResNet34(option='A').to(device)
        test_input = torch.rand(batch_size, 1, 224, 224).to(device)
        result = model.forward(test_input)
        shape = result.shape

        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], batch_size)
        self.assertEqual(shape[1], 1000)

        del model, test_input

    def test_residual_B(self):
        batch_size = 128
        model = models.ResNet34(option='B').to(device)
        test_input = torch.rand(batch_size, 1, 224, 224).to(device)
        result = model.forward(test_input)
        shape = result.shape

        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], batch_size)
        self.assertEqual(shape[1], 1000)

        del model, test_input


class TestDimensions18Layer(unittest.TestCase):
    def test_plain(self):
        batch_size = 128
        model = models.ResNet18(residual=False).to(device)
        test_input = torch.rand(batch_size, 1, 224, 224).to(device)
        result = model.forward(test_input)
        shape = result.shape

        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], batch_size)
        self.assertEqual(shape[1], 1000)

        del model, test_input

    def test_residual_A(self):
        batch_size = 256
        model = models.ResNet18(option='A').to(device)
        test_input = torch.rand(batch_size, 1, 224, 224).to(device)
        result = model.forward(test_input)
        shape = result.shape

        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], batch_size)
        self.assertEqual(shape[1], 1000)

        del model, test_input

    def test_residual_B(self):
        batch_size = 256
        model = models.ResNet18(option='B').to(device)
        test_input = torch.rand(batch_size, 1, 224, 224).to(device)
        result = model.forward(test_input)
        shape = result.shape

        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0], batch_size)
        self.assertEqual(shape[1], 1000)

        del model, test_input


if __name__ == '__main__':
    unittest.main(verbosity=2)
