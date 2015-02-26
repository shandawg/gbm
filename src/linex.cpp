// Linex loss function (LINEX): exp(a * (y - yhat)) - a * (y - yhat) - 1
// Gradient: -a * sum(exp(a * (y - yhat)) - 1)
// bestConstant: (1 / a) * log((1 / N) * sum(exp(a * y)))

#include "linex.h"
#include <math.h>
#include <iostream>

CLinex::CLinex()
{
}

CLinex::~CLinex()
{
}


GBMRESULT CLinex::ComputeWorkingResponse
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adF, 
    double *adZ, 
    double *adWeight,
    bool *afInBag,
    unsigned long nTrain,
	int cIdxOff
)
{
    GBMRESULT hr = GBM_OK;
    unsigned long i = 0;
	double dF = 0.0;

    if((adY == NULL) || (adF == NULL) || (adZ == NULL) || (adWeight == NULL))
    {
        hr = GBM_INVALIDARG;
        goto Error;
    }

	for(i=0; i<nTrain; i++)
	{

		dF = adF[i] + ((adOffset==NULL) ? 0.0 : adOffset[i]);
		// negative gradient (as stated in distribution.h)
        adZ[i] = 0.1 * (exp(0.1 * (adY[i] - dF)) - 1.0);
	}

Cleanup:
    return hr;
Error:
    goto Cleanup;
}



GBMRESULT CLinex::InitF
(
    double *adY,
    double *adMisc,
    double *adOffset, 
    double *adWeight,
    double &dInitF, 
    unsigned long cLength
)
{
    double dSum=0.0;
    double dTotalWeight = 0.0;
    unsigned long i=0;

    if(adOffset==NULL)
    {
        for(i=0; i<cLength; i++)
        {
            dSum += adWeight[i] * exp(0.1 * adY[i]);
            dTotalWeight += adWeight[i];
        }
    }
    else 
    {
        for(i=0; i<cLength; i++)
        {
        	dSum += adWeight[i] * exp(0.1 * (adY[i] - adOffset[i]));
            dTotalWeight += adWeight[i];
        }
    }

	dInitF = (1.0 / 0.1) * log(dSum / dTotalWeight);

    return GBM_OK;
}


double CLinex::Deviance
(
    double *adY,
    double *adMisc,
    double *adOffset, 
    double *adWeight,
    double *adF,
    unsigned long cLength,
	int cIdxOff
)
{
	unsigned long i=0;
	double dL = 0.0;
	double dW = 0.0;
	double dF = 0.0;

	for(i=cIdxOff; i<cLength+cIdxOff; i++)
	{
		dF = adF[i] + ((adOffset==NULL) ? 0.0 : adOffset[i]);
		dL += adWeight[i] * (exp(0.1 * (adY[i] - dF)) - 0.1 * (adY[i] - dF) - 1);
		dW += adWeight[i];
	}

	return dL/dW;
}


GBMRESULT CLinex::FitBestConstant
(
    double *adY,
    double *adMisc,
    double *adOffset,
    double *adW,
    double *adF,
    double *adZ,
    const std::vector<unsigned long>& aiNodeAssign,
    unsigned long nTrain,
    VEC_P_NODETERMINAL vecpTermNodes,
    unsigned long cTermNodes,
    unsigned long cMinObsInNode,
    bool *afInBag,
    double *adFadj,
	int cIdxOff
)
{
    GBMRESULT hr = GBM_OK;

	double dF = 0.0;
    unsigned long iObs = 0;
    unsigned long iNode = 0;
    unsigned long iVecd = 0;
    double dL;
    double dW;
    double dOffset;

    vecdNum.resize(cTermNodes);
    vecdNum.assign(vecdNum.size(),0.0);
    vecdDen.resize(cTermNodes);
    vecdDen.assign(vecdDen.size(),0.0);

    vecdMax.resize(cTermNodes);
    vecdMax.assign(vecdMax.size(),-HUGE_VAL);
    vecdMin.resize(cTermNodes);
    vecdMin.assign(vecdMin.size(),HUGE_VAL);

    std::vector<double> adW2(nTrain);

    for(iNode=0; iNode<cTermNodes; iNode++)
    {
        if(vecpTermNodes[iNode]->cN >= cMinObsInNode)
        {
            iVecd = 0;
            dL = 0.0;
            dW = 0.0;

            for(iObs=0; iObs<nTrain; iObs++)
            {
                if(afInBag[iObs] && (aiNodeAssign[iObs] == iNode))
                {
                    dOffset = (adOffset==NULL) ? 0.0 : adOffset[iObs];
                    dL += adW[iObs] * exp(0.1 * (adY[iObs] - dOffset -adF[iObs]));
                    dW += adW[iObs];
                    iVecd++;
                }
            }

            vecpTermNodes[iNode]->dPrediction = (1.0 / 0.1) * log((1.0 / dW) * dL);

        }
    }

    return hr;
}

double CLinex::BagImprovement
(
	double *adY,
	double *adMisc,
	double *adOffset,
	double *adWeight,
	double *adF,
	double *adFadj,
	bool *afInBag,
	double dStepSize,
	unsigned long nTrain
)
{
	double dReturnValue = 0.0;
	double dF = 0.0;
	double dW = 0.0;
	unsigned long i = 0;

	for(i=0; i<nTrain; i++)
	{
		if(!afInBag[i])
		{
			dF = adF[i] + ((adOffset==NULL) ? 0.0 : adOffset[i]);
			dReturnValue += adWeight[i] * (
				(exp(0.1 * (adY[i] - dF)) - 0.1 * (adY[i] - dF) - 1) - (exp(0.1 * (adY[i] - (dF + dStepSize * adFadj[i]))) - 0.1 * (adY[i] - (dF + dStepSize * adFadj[i])) - 1));
			dW += adWeight[i];
		}
	}

	return dReturnValue/dW;
}
