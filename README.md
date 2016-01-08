#Texture Synthesizer
	##Alex Kohler
A pixel based texture synthesis algorithm, based off of psuedocode from Efros and Leung's 1999 paper.

#USAGE (Octave/matlab) -
This was tested in Octave, however, instructions should be similar if not identical for matlab.
 Load required packages - pkg load image
 Source file        	- source growimage.m
 Call growimage         - e.g. growimage("sampleTextures/fire.JPG",5,20)
 For more information on parameters/individual functions, see function documentation.

Output filename is in format synthesized_<FILENAME>_winsize_<WINSIZE>_<OUTPUTSIZE>x<OUTPUTSIZE>. Currently input image must be in JPG or jpg format. I've included a couple 20x20 output images with varying window sizes an example in the synthesizedTextures directory. 

#References:
	http://graphics.cs.cmu.edu/people/efros/research/NPS/alg.html
	http://luthuli.cs.uiuc.edu/~daf/courses/CS-498-DAF-PS/Texture_498.pdf 
	http://eric-yuan.me/texture-synthesis/
	http://www.cs.umd.edu/~djacobs/CMSC426/PS5.doc
	http://www.ics.uci.edu/~dramanan/teaching/cs116_fall08/hw/Project/Texture/
