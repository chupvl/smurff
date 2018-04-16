#!/usr/bin/env python

from math import sqrt
import numpy as np
from scipy import sparse
import scipy.io as sio
import pandas as pd
import os
import os.path
import matrix_io as mio
from glob import glob
import re
import csv
import configparser

from .result import ResultItem

class OptionsFile(configparser.ConfigParser):
    def __init__(self, file_name):
        configparser.ConfigParser.__init__(self) 
        with open(file_name) as f:
            self.read_file(f, file_name)



class HeadlessConfigParser:
    def __init__(self, file_name):
        self.cp = configparser.ConfigParser()
        with  open(file_name) as f:
            content = "[top-level]\n" + f.read()
            self.cp.read_string(content)

    def __getitem__(self, key):
        return self.cp["top-level"][key]

    def items(self):
        return self.cp.items("top-level")


def make_abs(basedir, filename):
    if not os.path.isabs(filename):
        return os.path.join(basedir, filename)
    return filename

    
class TrainStep:
    @classmethod
    def fromStepFile(cls, basedir, file_name, iter):
        cp = HeadlessConfigParser(make_abs(basedir, file_name))
        step = cls(int(cp["num_models"]), iter)
        step.predictions = pd.read_csv(make_abs(basedir, cp["pred"]), sep=";")
        for i in range(step.nmodes):
            step.addU(mio.read_matrix(make_abs(basedir, cp["model_" + str(i)])))

        return step

    def __init__(self, nmodes, iter):
        assert nmodes == 2
        self.nmodes = nmodes
        self.iter = iter
        self.num_latent = None
        self.shape = []
        self.Us = []
        self.predictions = None

    def addU(self, U):
        self.Us.append(U)

        if self.num_latent is None:
            self.num_latent = U.shape[0]
        else:
            assert self.num_latent == U.shape[0]

        self.shape.append(U.shape[1])

    def predict_one(self, coords):
        return np.dot(self.Us[0][:,coords[0]], self.Us[1][:,coords[1]])

    def update_predictions(self, old_predictions):
        new_predictions = []

        for r in old_predictions:
            pred = self.predict_one(r.coords)
            delta = pred - r.pred_avg
            pred_avg = (r.pred_avg + delta / (self.iter + 1))
            var = r.var + delta * (pred - pred_avg)
            stds = float("nan") if self.iter == 0  else sqrt ( var / self.iter )
            r = ResultItem(r.coords, r.y, pred, pred_avg, var, stds)
            new_predictions.append(r)

        return new_predictions

    def predict_all(self):
        return self.Us[0] * self.Us[1]

class PredictSession:
    @classmethod
    def fromRootFile(cls, root_file):
        print("root_file = ", root_file)
        cp = HeadlessConfigParser(root_file)
        basedir = os.path.dirname(root_file)
        print("basedir = ", basedir)
        print("options = ", cp["options"])
        options = OptionsFile(make_abs(basedir, cp["options"]))

        session = cls(options.getint("global", "num_priors"))
        for step_name, step_file in cp.items():
            if (step_name.startswith("sample_step")):
                iter = int(step_name[len("sample_step_"):])
                session.addStep(TrainStep.fromStepFile(basedir, step_file, iter))

        assert len(session.steps) > 0

        return session

    def __init__(self, nmodes):
        assert nmodes == 2
        self.nmodes = nmodes
        self.num_latent = None
        self.shape = None

        self.steps = []

    def addStep(self, step):
        self.steps.append(step)

        if self.num_latent is None:
            self.num_latent = step.num_latent
        else:
            assert self.num_latent == step.num_latent

        if self.shape is None:
            self.shape = step.shape
        else:
            assert self.shape == step.shape
    
    def __str__(self):
        return "PredictSession with %d models\n  Data shape = %s\n  Num latent: %d" % (len(self.steps), self.shape, self.num_latent)