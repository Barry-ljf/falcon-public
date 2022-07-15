#pragma once
#include "AdaptAvgpoolLayer.h"
#include "Functionalities.h"
using namespace std;


AdaptAvgpoolLayer::AdaptAvgpoolLayer(AdaptAvgpoolLayerConfig* conf, int _layerNum)
:Layer(_layerNum),
 conf(conf->imageHeight, conf->imageWidth, conf->features, 
	  conf->poolSize, conf->stride, conf->batchSize),
 activations(conf->batchSize * conf->features * 
		    (((conf->imageWidth - conf->poolSize)/conf->stride) + 1) * 
 		    (((conf->imageHeight - conf->poolSize)/conf->stride) + 1)),
 deltas(conf->batchSize * conf->features * 
	   (((conf->imageWidth - conf->poolSize)/conf->stride) + 1) * 
	   (((conf->imageHeight - conf->poolSize)/conf->stride) + 1)),
 maxPrime((((conf->imageWidth - conf->poolSize)/conf->stride) + 1) * 
		 (((conf->imageHeight - conf->poolSize)/conf->stride) + 1) * 
		 conf->features * conf->batchSize * conf->poolSize * conf->poolSize)
{};


void AdaptAvgpoolLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;  	
	cout << "(" << layerNum+1 << ") AdaptAvgpool Layer\t  " << conf.imageHeight << " x " << conf.imageWidth 
		 << " x " << conf.features << endl << "\t\t\t  " 
		 << conf.poolSize << "  \t\t(Pooling Size)" << endl << "\t\t\t  " 
		 << conf.stride << " \t\t(Stride)" << endl << "\t\t\t  " 
		 << conf.batchSize << "\t\t(Batch Size)" << endl;
}

void AdaptAvgpoolLayer::forward(const RSSVectorMyType& inputActivation)
{
	log_print("AdaptAvgpool.forward");
    //now we want to do is to make n*n features to 1*1.
    //pool size  = (iw + 2 * pading) - (ow - 1) * S
    //pool size = iw - (ow - 1) * S
    //because we want ow to be 1 so we get pool_size = iw
	size_t B 	= conf.batchSize;
	size_t iw 	= conf.imageWidth;
	size_t ih 	= conf.imageHeight;
	size_t f 	= conf.poolSize;
	size_t Din 	= conf.features;
	size_t S 	= conf.stride;
	size_t ow 	= 1;
	size_t oh	= 1;

	RSSVectorMyType temp1(ow*oh*Din*B*f*f);
	{
		size_t sizeBeta = iw;
		size_t sizeD 	= sizeBeta*ih;
		size_t sizeB 	= sizeD*Din;
		size_t counter 	= 0;
		for (int b = 0; b < B; ++b)
			for (size_t r = 0; r < Din; ++r)
				for (size_t beta = 0; beta < ih-f+1; beta+=S) 
					for (size_t alpha = 0; alpha < iw-f+1; alpha+=S)
						for (int q = 0; q < f; ++q)
							for (int p = 0; p < f; ++p)
							{
								temp1[counter++] = 
									inputActivation[b*sizeB + r*sizeD + 
										(beta + q)*sizeBeta + (alpha + p)];
							}
	}
    //temp1 is to rebuild input structure

	//Pooling operation
	if (FUNCTION_TIME)
		cout << "funcAdaptAvgpool: " << funcTime(funcAdaptAvgpool, temp1, activations, ow*oh*Din*B, f*f) << endl;
	else
		funcAdaptAvgpool(temp1, activations,  ow*oh*Din*B, f*f);
        //ow*oh*Din*B is an output of layer ,every single bits multiply (f*f) get the original size.
		//activations's size is equal to output
}


void AdaptAvgpoolLayer::computeDelta(RSSVectorMyType& prevDelta)
{
	log_print("AdaptAvgpoolLayer.computeDelta");

}

void AdaptAvgpoolLayer::updateEquations(const RSSVectorMyType& prevActivations)
{
	log_print("AdaptAvgpool.updateEquations");
}
