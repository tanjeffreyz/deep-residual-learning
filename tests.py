import unittest
import torch
import models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[~] Using {device}')


def dimension_test_case(model, n, in_dims: tuple, out_size, batch_size=128):
    def case(residual, option):
        def test(self):
            m = model(n, residual=residual, option=option).to(device)
            test_input = torch.rand(batch_size, in_dims[0], in_dims[1], in_dims[1]).to(device)
            result = m.forward(test_input)
            shape = result.shape

            self.assertEqual(len(shape), 2)
            self.assertEqual(shape[0], batch_size)
            self.assertEqual(shape[1], out_size)

            del m, test_input

        return test

    return type(f'TestDimensions{model.__name__}{n}', (unittest.TestCase,), {
        'test_plain': case(False, None),
        'test_residual_A': case(True, 'A'),
        'test_residual_B': case(True, 'B')
    })


TestDimensionsCifar20 = dimension_test_case(models.CifarResNet, 20, (3, 32), 10)
TestDimensionsCifar32 = dimension_test_case(models.CifarResNet, 32, (3, 32), 10)
TestDimensionsCifar44 = dimension_test_case(models.CifarResNet, 44, (3, 32), 10)
TestDimensionsCifar56 = dimension_test_case(models.CifarResNet, 56, (3, 32), 10)
TestDimensionsCifar110 = dimension_test_case(models.CifarResNet, 110, (3, 32), 10)

TestDimensionsImageNet18 = dimension_test_case(models.ImageNetResNet, 18, (1, 224), 1000)
TestDimensionsImageNet34 = dimension_test_case(models.ImageNetResNet, 34, (1, 224), 1000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
