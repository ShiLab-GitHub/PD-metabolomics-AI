import sys
from RF_RFE import rf
from XGB_RFE import xgb
from GBM_RFE import gbm

if __name__ == '__main__':
    model = sys.argv[1]
    n_classes = sys.argv[2]
    if model == "RF-RFE":
        rf(n_classes)
    if model == "XGB-RFE":
        xgb(n_classes)
    if model == "GBM-RFE":
        gbm(n_classes)