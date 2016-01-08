function growimage(textureFileName, winSize, outputSize)
%  function [outputImage] = growimage(textureFileName, winSize, outputSize)
%  Given an input image, returns a new outputSizexOutputsize image grown outward from an initial seed from the input image, one pixel at a time.  
%
%  	textureFileName = path of input texture image (filename is in format synthesized_<FILENAME>_winsize_<WINSIZE>_<OUTPUTSIZE>x<OUTPUTSIZE>. Currently input image must be in JPG or jpg format
%  	winSize = texton neighborhood diameter. MUST BE ODD
%	outputSize = edge length of output image (square) 
%
%   Based off of psueodocode from Efros and Leung (1999). For more information, see http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html    
%   See README for other references.

	% Check for valid winSize 
    if ~mod(winSize,2) 
		disp('ERROR: Invalid parameter winSize: Please enter an odd neighborhood diameter')
		break
	end

    % Initialize Sigma and error thresholds based off of recommendations in original implementation
    Sigma = winSize/6.4;
    ErrThreshold = 0.1;
    MaxErrThreshold = 0.3;
    
    % read in texture
    rawSample = im2double(imread(textureFileName));
	% colorChannels - 1 for a gray scale image, 3 for an RGB image or volumetric gray scale image, and >3 for multi-spectral images or higher dimensional arrays. For our sake, we will only consider RGB and will hence disregard this third parameter
    [rows, cols, ~] = size(rawSample);

	% Initialize the template we will be writing to, expanding by outputSize in x/y, will always assume we have 3 colors (no grayscale) 
    texture = zeros(rows+outputSize,cols+outputSize,3);
    texture(1:rows,1:cols,:) = rawSample;
	
	% grab final dimensions
    tex_rows = size(texture,1);
    tex_cols = size(texture,2);

	% current/final sizes
    currentlyFilled = rows * cols;
    filledImageSize = tex_rows * tex_cols;
	
	% Keep track of which pixels we've written onto (Written - true, unwritten - false)
    explorationMap = false(tex_rows,tex_cols);
    explorationMap(1:rows,1:cols) = true([rows cols]);
    
    while currentlyFilled < filledImageSize
        progress = 0;
        pixelList = getUnfilledNeighbors(explorationMap,winSize);

        for i = pixelList;
			pixRow = i(1);
			pixCol = i(2);
            [template, validMask] = getNeighborhoodWindow(winSize,texture,explorationMap,pixRow,pixCol);
            [bestMatches, SSD] = findMatch(Sigma,winSize,template, validMask, ErrThreshold, rawSample);
			[bestMatch, bestMatchVal, bestMatch_err] =  RandomPick(bestMatches, SSD, winSize, rawSample);
            if (bestMatch_err < MaxErrThreshold)
                texture(pixRow,pixCol,:) = bestMatchVal;
                explorationMap(pixRow, pixCol) = true;
                currentlyFilled = currentlyFilled + 1;
                progress = 1;
            end
        end
        
		% actively draw to show iterative progress
        imshow(texture);
        drawnow;
        
        if (progress==0)
            MaxErrThreshold = MaxErrThreshold * 1.1;
        end
    end
    
    imshow(texture);
    outputImage = texture;
	% remove unneeded fluff (path, file extension) from our output filename
	outputFileName=regexprep(textureFileName,'sampleTextures/','','once');
	outputFileName=regexprep(outputFileName,'.jpg|.JPG','','once');
	% create final final name
	outputFileName = strcat('synth_', outputFileName, '_winsize_', int2str(winSize), '_', int2str(outputSize), 'x', int2str(outputSize), '.jpg');
	S=sprintf('Wrote syntheisized texture to %s', outputFileName);
	disp(S)
	imwrite(outputImage, outputFileName, 'jpg');
end

function [PixelList] = getUnfilledNeighbors(explorationMap,winSize)
% getUnfilledNeigbors - returns a list of all unfilled pixels that have filled pixels as their neighbors (the image is subtracted from its morphological dilation). The list is randomly permuted and then sorted by decreasing number of filled neighbor pixels. (helper function)

	% expand by one in every direction
    morphDilation = imdilate(explorationMap, ones(3));
	% image is subtracted from its morphological dilation	
    unfilledPixels = morphDilation - explorationMap > 0;
	
    [pixelRows pixelCols] = find(unfilledPixels);
    
	%randomly permute
    randIndex = randperm(length(pixelRows));
    pixelRows = pixelRows(randIndex);
    pixelCols = pixelCols(randIndex);
    
    neighSums = colfilt(explorationMap,[winSize winSize],'sliding',@sum);
    
    linearIndex = sub2ind(size(neighSums),pixelRows,pixelCols);
	% sort max amount of neighbors in descending order
    [~, index] = sort(neighSums(linearIndex),'descend');
    sorted = linearIndex(index);
    [pixelRows, pixelCols] = ind2sub(size(explorationMap),sorted);
	PixelList = [pixelRows pixelCols]';
end

function [bestMatch, bestMatchVal, bestMatchErr] = RandomPick(bestMatches, SSD, winSize, rawSample)
	% RandomPick - picks an element randomly from the list of best matches, then computes its value and error (helper function)
    bestMatch = bestMatches(ceil(rand*length(bestMatches)));
    bestMatchErr = SSD(bestMatch);
	% grab equivalent subscript values
    [rowMatch, colMatch] = ind2sub(abs(size(rawSample) - winSize + 1), bestMatch);
    mid = floor((winSize - 1) / 2);
	% offset matches
   	bestMatchVal = rawSample(round(rowMatch + mid), round(colMatch + mid)  ,:);
end

function [template, validMask] = getNeighborhoodWindow(winSize,texture, explorationMap, pixelRow, pixelCol)
	% GetNeigborhoodWindow - returns a window of size WindowSize around a given pixel (retrieves validMask here instead of FindMatches as outlined in the Efros/Leung's pseudcode.) (helper function)

    mid = floor((winSize - 1) / 2);
	outputRow = size(texture,1);
	outputCol = size(texture,2);
	rLow = pixelRow - mid;
	rHigh = pixelRow + mid;
	row_range = rLow:rHigh;

	cLow = pixelCol - mid;
	cHigh = pixelCol + mid;
	column_range = cLow : cHigh;

	% accumulate invalid indices(out of bounds above or below)
	row_invHigh = row_range > outputRow;
	row_invLow =  row_range < 1;
	col_invHigh = column_range > outputCol;
	col_invLow =  column_range < 1;

	badRow = row_invLow | row_invHigh;	
	badCol = col_invLow | col_invHigh;

	rCont1 = any(badRow(:)==1);
	cCont1 = any(badCol(:)==1);
    if (rCont1  || cCont1) 
        goodRow = row_range(~badRow);
        goodCol = column_range(~badCol);

        template = zeros(winSize, winSize, 3);
        template(~badRow, ~badCol, :) = texture(goodRow, goodCol, :);

        validMask = false([winSize winSize]);
        validMask(~badRow, ~badCol) = explorationMap(goodRow, goodCol);
    else
		% fix off by one
		if (column_range(3) > size(explorationMap, 2))
			column_range(3)--;
		end
        template = texture(row_range, column_range, :);
        validMask = explorationMap(rLow:rHigh, cLow:cHigh);
    end

end

function [PixelList, SSD] = findMatch(Sigma,winSize,template, validMask, error_threshold, rawSample)
	% findMatch - Returns a list of possible candidate matches in the source texture using Gaussian SSD to ensure agreement along with their corresponding SSD errors. (helper function)

	% create 2D gaussian mask
    gaussMask = fspecial('gaussian',winSize,Sigma);
	% use suggested image mask
	mask = gaussMask .* validMask;
    weightTotal = sum(sum(mask));

	% image blocks to columns
    red_col = im2col(rawSample(:, :, 1), [winSize winSize]); 
    green_col = im2col(rawSample(:, :, 2), [winSize winSize]); 
    blue_col = im2col(rawSample(:, :, 3), [winSize winSize]);
	% we can avoid looping here! - http://www.cs.umd.edu/~djacobs/CMSC426/PS5.doc
    [pixelsInNeighborhood, numNeighborhoods] = size(red_col);
    
    red_vals = template(:,:,1);
    green_vals = template(:,:,2);
    blue_vals = template(:,:,3);

	% repeat copies of our RGB vals 
    red_vals = repmat(red_vals(:), [1 numNeighborhoods]); 
    green_vals = repmat(green_vals(:), [1 numNeighborhoods]); 
    blue_vals = repmat(blue_vals(:), [1 numNeighborhoods]); 
  
	% Gaussian weighted distances for our RGB vals 
    red_dist =  mask(:)' * (red_vals - red_col).^2; 
    green_dist = mask(:)' * (green_vals - green_col).^2; 
    blue_dist = mask(:)' * (blue_vals - blue_col).^2; 

    SSD = (red_dist + green_dist + blue_dist) / weightTotal; 
	
	% remove distances above error threshold
    PixelList = find(SSD <= min(SSD) .* (1+error_threshold));
end

