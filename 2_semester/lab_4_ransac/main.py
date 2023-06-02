import matplotlib.pyplot as plt

from data import PointGenerator
from ransac import RANSAC


def main():
    generator = PointGenerator(100, 0.1)
    points = generator.generate_case(k=1.5, b=2.0, eps=0.3)
    ransac = RANSAC()
    ransac.set_case(points, iter_num=1000, epsilon=0.35)
    params = ransac.fit()
    save_path = "example.png"
    print("Estimated line parameters:")
    print("\tLine equation: y={:2.4f}*x{:+2.4f}".format(params['k'], params['b']))
    print("Output saved by path:", save_path)
    ransac.draw(save_path=save_path)

if __name__ == "__main__":
    main()