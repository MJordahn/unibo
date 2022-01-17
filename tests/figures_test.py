from os import mkdir
import unittest
from main import *
from visualizations.scripts.ranking import Ranking
from visualizations.scripts.tables import Tables
from visualizations.scripts.figures import Figures


class ResultsTest(unittest.TestCase):
    def test_plots_default(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/tests/")
        loadpths = [os.getcwd() + "/results/tests/" + f + "/" for f in loadpths]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Figures(loadpths).generate()

    def test_tables_default(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/tests/")
        loadpths = [os.getcwd() + "/results/tests/" + f + "/" for f in loadpths]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Tables(loadpths, settings={"bo": True}).generate()
        Tables(loadpths, settings={"bo": False}).generate()

    def test_plots_epochs(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Figures(loadpths).calibration_vs_epochs()

    def test_ranking(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        Ranking(loadpths).run()

    def test_plots_bo_2d_contour(self) -> None:
        loadpths = os.listdir(os.getcwd() + "/results/")
        loadpths = [
            os.getcwd() + "/results/" + f + "/" for f in loadpths if "tests" not in f
        ]
        loadpths = [f for f in loadpths if os.path.isdir(f)]
        figures = Figures(loadpths)
        figures.bo_2d_contour(n_epochs=20, seed=2)
        # figures.bo_2d_contour(n_epochs=10, seed=1)


if __name__ == "__main__":
    unittest.main()
