

# Design idea:

	We have three components: 
		Video extractor
		Resnet
		RNN
		
	These should all be modular for reasons which I think are obvious. 
	
	
Now, all three should be their own classes so that we can mix and match them as we please depending on the need. 
That is, if we have an opencv stream then that should work, if we have a stream of pure PIL images then that should work as well. 


# RNN

My idea is that we use some kind of trained RNN *or* a pre-trained RNN. If we make it modular than both could work.


## Training 
	
	We construct an RNN that recives series of data that have been encoded by the Resnet. 
	This is then given in batches to the RNN which is supposed to predict a score. 
	We somehow need to normalize score between different datasets. My idea at the moment is to normalize it so that we can just use a 0-1 scale. Where 0 is bad quality and 1 is good. 
	Or if we want a wider space we can use a 0-10 scale.
	
	### Saving of model
	
		We need to make sure that the model is saved regularly so that we can use it after training!

## Evaluation
	
	We pick the type of model and which epoch (or something of that sort) and then we load the model. One idea is that the configuration itself is the filename, so that we know which model we pick.


# Video extractor

## Training 

	In this mode we already have all of the data. 
	This means that we do *not* need to fetch information one at a time.

## Live running


	Here we need to continously sample frames from some live video stream and just feed it to the model.
	
	
	
# Resnet

	The idea of the resnet is that it should be a stable-premade solution that we do not need to care about. We just specify which resnet we want to use and it solves the encoding etc itself. 

