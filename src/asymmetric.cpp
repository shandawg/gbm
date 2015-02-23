// Tweedie distribution with natural log link function ( mean = exp(prediction) )
// -19 <= prediction <= +19

// Parameter dPower defaults to 1.5

#include "tweedie.h"
#include <math.h>
#include <typeinfo>
#include <iostream>
#include <vector>
#include <deque>
#include <fstream>


CTweedie::~CTweedie()
{
}

GBMRESULT CTweedie::ComputeWorkingResponse
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
		adZ[i] = adY[i]*exp(dF*(1.0-dPower)) - exp(dF*(2.0-dPower));
    }

Cleanup:
    return hr;
Error:
    goto Cleanup;
}



GBMRESULT CTweedie::InitF
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
	double Min = -19.0;
	double Max = +19.0;
    unsigned long i=0;

	if(adOffset==NULL)
	{
		for(i=0; i<cLength; i++)
		{
			dSum += adWeight[i]*adY[i];
			dTotalWeight += adWeight[i];
		}
	}
	else
	{
		for(i=0; i<cLength; i++)
		{
			dSum += adWeight[i]*adY[i]*exp(adOffset[i]*(1.0-dPower));
			dTotalWeight += adWeight[i]*exp(adOffset[i]*(2.0-dPower));
		}
	}

	if (dSum <= 0.0) { dInitF = Min; }
	else { dInitF = log(dSum/dTotalWeight); }

	if (dInitF < Min) { dInitF = Min; }
	if (dInitF > Max) { dInitF = Max; }

	return GBM_OK;
}


double CTweedie::Deviance
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
	double dF = 0.0;
	unsigned long i=0;
	double dL = 0.0;
	double dW = 0.0;

	for(i=cIdxOff; i<cLength+cIdxOff; i++)
	{
		dF = adF[i] + ((adOffset==NULL) ? 0.0 : adOffset[i]);
		dL += adWeight[i]*(pow(adY[i],2.0-dPower)/((1.0-dPower)*(2.0-dPower)) -
			adY[i]*exp(dF*(1.0-dPower))/(1.0-dPower) + exp(dF*(2.0-dPower))/(2.0-dPower) );
		dW += adWeight[i];
	}

	return 2.0*dL/dW;
}


GBMRESULT CTweedie::FitBestConstant
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
	double MaxVal = 19.0;
	double MinVal = -19.0;
	
    vecdNum.resize(cTermNodes);
    vecdNum.assign(vecdNum.size(),0.0);
    vecdDen.resize(cTermNodes);
    vecdDen.assign(vecdDen.size(),0.0);

    vecdMax.resize(cTermNodes);
    vecdMax.assign(vecdMax.size(),-HUGE_VAL);
    vecdMin.resize(cTermNodes);
    vecdMin.assign(vecdMin.size(),HUGE_VAL);

	for(iObs=0; iObs<nTrain; iObs++)
	{
		if(afInBag[iObs])
		{
			dF = adF[iObs] + ((adOffset==NULL) ? 0.0 : adOffset[iObs]);
			vecdNum[aiNodeAssign[iObs]] += adW[iObs]*adY[iObs]*exp(dF*(1.0-dPower));
			vecdDen[aiNodeAssign[iObs]] += adW[iObs]*exp(dF*(2.0-dPower));

			// Keep track of largest and smallest prediction in each node
			vecdMax[aiNodeAssign[iObs]] = fmax2(dF,vecdMax[aiNodeAssign[iObs]]);
			vecdMin[aiNodeAssign[iObs]] = fmin2(dF,vecdMin[aiNodeAssign[iObs]]);
		}
	}

	for(iNode=0; iNode<cTermNodes; iNode++)
	{
		if(vecpTermNodes[iNode]!=NULL)
		{
			if(vecdNum[iNode] == 0.0)
			{
				// Taken from poisson.cpp

				// DEBUG: if vecdNum==0 then prediction = -Inf
				// Not sure what else to do except plug in an arbitrary
				//   negative number, -1? -10? Let's use -19, then make
				//   sure |adF| < 19 always.
				vecpTermNodes[iNode]->dPrediction = MinVal;
			}

			else if(vecdDen[iNode] == 0.0) { vecpTermNodes[iNode]->dPrediction = 0.0; }

			else { vecpTermNodes[iNode]->dPrediction = log(vecdNum[iNode]/vecdDen[iNode]); }

			if (vecdMax[iNode]+vecpTermNodes[iNode]->dPrediction > MaxVal)
				{vecpTermNodes[iNode]->dPrediction = MaxVal - vecdMax[iNode];}
			if (vecdMin[iNode]+vecpTermNodes[iNode]->dPrediction < MinVal)
				{vecpTermNodes[iNode]->dPrediction = MinVal - vecdMin[iNode];}

		}
	}

	return hr;
}

double CTweedie::BagImprovement
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

			dReturnValue += adWeight[i]*( exp(dF*(1.0-dPower))*adY[i]/(1.0-dPower)*
				(exp(dStepSize*adFadj[i]*(1.0-dPower))-1.0) +
				exp(dF*(2.0-dPower))/(2.0-dPower)*(1.0-exp(dStepSize*adFadj[i]*(2.0-dPower))) );
			dW += adWeight[i];
		}
	}

	return 2.0*dReturnValue/dW;
}