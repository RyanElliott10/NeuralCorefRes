# -*- coding: utf-8 -*-
# Simple flask server to decrease data preprocessing times.
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

import os
import sys

from flask import Flask, jsonify

sys.path.append(os.path.abspath(f'{os.path.dirname(os.path.abspath(__file__))}/../'))
from neuralcorefres.model.coreference_network import CoreferenceNetwork
from neuralcorefres.parsedata.parse_clusters import ParseClusters

app = Flask(__name__)


class Facade:
    inst = None

    @staticmethod
    def get_instance():
        if Facade.inst is None:
            Facade.inst = Facade()
        return Facade.inst

    def __init__(self):
        self._parse_data()

    def _parse_data(self):
        self.sents, self.clusters = ParseClusters.get_from_file('../data/PreCo_1.0/custom_dps/dev.json')
        self.x_train, self.y_train = CoreferenceNetwork.custom_cluster_to_nn_input(self.sents[:1], self.clusters[:1])
        self.x_train_list = self.x_train.tolist()[:1]
        self.y_train_list = self.y_train.tolist()[:1]
        self.json = jsonify({'x_train': self.x_train_list, 'y_train': self.y_train_list})


@app.route('/data')
def get_ep():
    inst = Facade.get_instance()
    return inst.json


@app.route('/')
def hello_world():
    return jsonify({'key': ['value', 'hello world']})


if __name__ == '__main__':
    app.run()
