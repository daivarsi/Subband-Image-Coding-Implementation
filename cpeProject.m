% Multi-Level Wavelet Decomposition and Compression

% Specify the full path to the input image
color = 1;

if color == 0
    input_image_path = 'C:\Users\DSM\Desktop\cpeProject\lena_gray.bmp';
else
    input_image_path = 'C:\Users\DSM\Desktop\cpeProject\lena_color.tiff';
end


% Read the input image
original_image = imread(input_image_path);

% Convert to double for processing
original_image = double(original_image);

% Decomposition levels
decomposition_levels = 3;

% Compression parameters
quantization_steps = [5, 10, 15, 20]; % Different compression levels

% Process for different quantization steps
for i = 1:length(quantization_steps)
    % Current quantization step
    step_size = quantization_steps(i);

    % Process each color channel separately
    num_channels = size(original_image, 3);
    compressed_channels = cell(1, num_channels);
    reconstructed_channels = cell(1, num_channels);
    channel_sizes = cell(1, num_channels);

    for channel = 1:num_channels
        % Extract current channel
        current_channel = original_image(:,:,channel);
        
        % Multi-level Wavelet Decomposition
        [coeffs, sizes] = multi_level_wavelet_decomposition(current_channel, decomposition_levels);
        
        % Scalar Quantization of coefficients
        quantized_coeffs = cell(size(coeffs));
        for j = 1:length(coeffs)
            quantized_coeffs{j} = round(coeffs{j} / step_size);
        end
        
        % Entropy Encoding (Huffman)
        encoded_data = cell(size(coeffs));
        huffman_tables = cell(size(coeffs));
        compressed_sizes = zeros(size(coeffs));
        
        for j = 1:length(quantized_coeffs)
            [encoded_data{j}, huffman_tables{j}] = huffman_encode(quantized_coeffs{j});
            compressed_sizes(j) = numel(encoded_data{j});
        end
        
        % Store compressed channel data
        compressed_channels{channel} = encoded_data;
        channel_sizes{channel} = sizes;
        
        % Huffman decoding
        decoded_coeffs = cell(size(coeffs));
        for j = 1:length(encoded_data)
            decoded_coeffs{j} = huffman_decode(encoded_data{j}, huffman_tables{j}, ...
                size(quantized_coeffs{j}));
        end
        
        % Dequantization
        dequantized_coeffs = cell(size(decoded_coeffs));
        for j = 1:length(decoded_coeffs)
            dequantized_coeffs{j} = decoded_coeffs{j} * step_size;
        end
        
        % Reconstruct Channel
        reconstructed_channel = multi_level_wavelet_reconstruction(dequantized_coeffs, sizes, decomposition_levels);
        reconstructed_channels{channel} = max(min(round(reconstructed_channel), 255), 0);
    end
    
    % Combine reconstructed channels
    reconstructed_image = zeros(size(original_image));
    for channel = 1:num_channels
        reconstructed_image(:,:,channel) = reconstructed_channels{channel};
    end
    
    % Save the compressed data
    [folder, name, ~] = fileparts(input_image_path);
    compressed_filename = fullfile(folder, sprintf('%s_compressed_step_%d.bin', name, step_size));
    
    % Save compressed data with color channel information
    save_color_compressed_data(compressed_filename, compressed_channels, huffman_tables, ...
        channel_sizes, decomposition_levels, step_size, size(original_image));
        
    % Calculate PSNR for each channel and average
    psnr_values = zeros(1, num_channels);
    for channel = 1:num_channels
        psnr_values(channel) = calculate_psnr(original_image(:,:,channel), ...
            reconstructed_image(:,:,channel));
    end
    psnr_value = mean(psnr_values);
    
    % Calculate Compression Ratio
    original_size_bits = numel(original_image) * 8;
    compressed_size_bits = sum(compressed_sizes) * num_channels;
    compression_ratio = original_size_bits / compressed_size_bits;
    
    % Save reconstructed image
    filename = fullfile(folder, sprintf('%s_reconstructed_step_%d.png', name, step_size));
    imwrite(uint8(reconstructed_image), filename);
    
    % Print compression details
    fprintf('Quantization Step: %d\n', step_size);
    fprintf('Decomposition Levels: %d\n', decomposition_levels);
    fprintf('Average PSNR: %.2f dB\n', psnr_value);
    fprintf('Individual Channel PSNRs: R=%.2f, G=%.2f, B=%.2f dB\n', psnr_values);
    fprintf('Compression Ratio: %.2f:1\n', compression_ratio);
    fprintf('Original Size: %.2f KB\n', original_size_bits/8/1024);
    fprintf('Compressed Size: %.2f KB\n', compressed_size_bits/8/1024);
    fprintf('Reconstructed Image Saved: %s\n\n', filename);
end

% Visualization of results
figure;
subplot(2,1,1);
plot(quantization_steps, calculate_psnr_values(original_image, quantization_steps, decomposition_levels), '-o');
title('PSNR vs Quantization Step Size');
xlabel('Quantization Step Size');
ylabel('PSNR (dB)');

subplot(2,1,2);
plot(quantization_steps, calculate_compression_ratios(original_image, quantization_steps, decomposition_levels), '-o');
title('Compression Ratio vs Quantization Step Size');
xlabel('Quantization Step Size');
ylabel('Compression Ratio');

function [coeffs, sizes] = multi_level_wavelet_decomposition(image, levels)
    % Initialize coefficients storage
    coeffs = cell(1, levels * 3 + 1);
    sizes = cell(1, levels + 1);
    
    % Current image for decomposition
    current_image = image;
    
    % Store original image size
    sizes{1} = size(current_image);
    
    % Perform multi-level decomposition
    for level = 1:levels
        % Ensure even dimensions
        [rows, cols] = size(current_image);
        rows = floor(rows/2)*2;
        cols = floor(cols/2)*2;
        current_image = current_image(1:rows, 1:cols);
        
        % Perform 2D wavelet decomposition
        [LL, LH, HL, HH] = perform_2d_wavelet_decomposition(current_image);
        
        % Store coefficient matrices
        coeffs{level*3-2} = LH;
        coeffs{level*3-1} = HL;
        coeffs{level*3} = HH;
        
        % Store size of current decomposition
        sizes{level + 1} = size(LL);
        
        % Continue decomposition on LL band
        current_image = LL;
    end
    
    % Store the final LL band
    coeffs{end} = current_image;
end

function [LL, LH, HL, HH] = perform_2d_wavelet_decomposition(image)
    % Get dimensions
    [rows, cols] = size(image);
    
    % Ensure even dimensions
    rows = floor(rows/2)*2;
    cols = floor(cols/2)*2;
    image = image(1:rows, 1:cols);
    
    % Initialize subbands
    LL = zeros(rows/2, cols/2);
    LH = zeros(rows/2, cols/2);
    HL = zeros(rows/2, cols/2);
    HH = zeros(rows/2, cols/2);
    
    % Process rows
    temp_L = zeros(rows, cols/2);
    temp_H = zeros(rows, cols/2);
    
    for i = 1:rows
        for j = 1:cols/2
            temp_L(i,j) = (image(i,2*j-1) + image(i,2*j))/sqrt(2);
            temp_H(i,j) = (image(i,2*j-1) - image(i,2*j))/sqrt(2);
        end
    end
    
    % Process columns
    for j = 1:cols/2
        for i = 1:rows/2
            LL(i,j) = (temp_L(2*i-1,j) + temp_L(2*i,j))/sqrt(2);
            LH(i,j) = (temp_L(2*i-1,j) - temp_L(2*i,j))/sqrt(2);
            HL(i,j) = (temp_H(2*i-1,j) + temp_H(2*i,j))/sqrt(2);
            HH(i,j) = (temp_H(2*i-1,j) - temp_H(2*i,j))/sqrt(2);
        end
    end
end

function reconstructed_image = multi_level_wavelet_reconstruction(coeffs, sizes, levels)
    % Start with the lowest frequency band
    current_image = coeffs{end};
    
    % Reconstruct from lowest to highest frequency
    for level = levels:-1:1
        % Get detail coefficients for current level
        LH = coeffs{level*3-2};
        HL = coeffs{level*3-1};
        HH = coeffs{level*3};
        
        % Verify all subbands have the same size
        if ~isequal(size(current_image), size(LH), size(HL), size(HH))
            % Resize all subbands to match the expected size from sizes array
            target_size = sizes{level};
            current_image = imresize(current_image, target_size);
            LH = imresize(LH, target_size);
            HL = imresize(HL, target_size);
            HH = imresize(HH, target_size);
        end
        
        % Perform inverse wavelet transform
        current_image = perform_2d_inverse_wavelet_reconstruction(current_image, LH, HL, HH);
    end
    
    % Final resize to original image size if needed
    if ~isequal(size(current_image), sizes{1})
        reconstructed_image = imresize(current_image, sizes{1});
    else
        reconstructed_image = current_image;
    end
end

function reconstructed = perform_2d_inverse_wavelet_reconstruction(LL, LH, HL, HH)
    % Get dimensions from input
    [rows, cols] = size(LL);
    
    % Pre-allocate arrays with correct sizes
    temp_L = zeros(2*rows, cols);
    temp_H = zeros(2*rows, cols);
    reconstructed = zeros(2*rows, 2*cols);
    
    % Process columns - inverse vertical transform
    for j = 1:cols
        for i = 1:rows
            % Ensure indices are within bounds
            idx1 = min(2*i-1, 2*rows);
            idx2 = min(2*i, 2*rows);
            
            % Apply inverse transform
            temp_L(idx1,j) = (LL(i,j) + LH(i,j))/sqrt(2);
            temp_L(idx2,j) = (LL(i,j) - LH(i,j))/sqrt(2);
            temp_H(idx1,j) = (HL(i,j) + HH(i,j))/sqrt(2);
            temp_H(idx2,j) = (HL(i,j) - HH(i,j))/sqrt(2);
        end
    end
    
    % Process rows - inverse horizontal transform
    for i = 1:2*rows
        for j = 1:cols
            % Ensure indices are within bounds
            idx1 = min(2*j-1, 2*cols);
            idx2 = min(2*j, 2*cols);
            
            % Apply inverse transform
            reconstructed(i,idx1) = (temp_L(i,j) + temp_H(i,j))/sqrt(2);
            reconstructed(i,idx2) = (temp_L(i,j) - temp_H(i,j))/sqrt(2);
        end
    end
end

function [encoded_data, huffman_table] = huffman_encode(data)
    % Convert input data to a column vector of integers
    data_vector = round(data(:));
    
    % Get unique values and their frequencies
    unique_vals = unique(data_vector);
    freq = histcounts(data_vector, [unique_vals; max(unique_vals)+1]);
    
    % Create Huffman dictionary
    p = freq/sum(freq);
    dict = huffmandict(unique_vals, p);
    
    % Encode the data
    encoded_data = huffmanenco(data_vector, dict);
    
    % Store the Huffman table for decoding
    huffman_table = dict;
end

function decoded_data = huffman_decode(encoded_data, huffman_table, target_size)
    % Decode the data
    decoded_vector = huffmandeco(encoded_data, huffman_table);
    
    % Reshape to original dimensions
    decoded_data = reshape(decoded_vector, target_size);
end

% Added helper function to store sizes in metadata

function psnr_value = calculate_psnr(original, reconstructed)
    mse = mean((original(:) - reconstructed(:)).^2);
    max_pixel = max(original(:));
    psnr_value = 10 * log10((max_pixel^2) / mse);
end

function psnr_values = calculate_psnr_values(original_image, quantization_steps, decomposition_levels)
    psnr_values = zeros(size(quantization_steps));
    
    for i = 1:length(quantization_steps)
        step_size = quantization_steps(i);
        
        % Perform decomposition
        [coeffs, sizes] = multi_level_wavelet_decomposition(original_image, decomposition_levels);
        
        % Quantize
        quantized_coeffs = cell(size(coeffs));
        for j = 1:length(coeffs)
            quantized_coeffs{j} = round(coeffs{j} / step_size);
        end
        
        % Dequantize
        dequantized_coeffs = cell(size(quantized_coeffs));
        for j = 1:length(quantized_coeffs)
            dequantized_coeffs{j} = quantized_coeffs{j} * step_size;
        end
        
        % Reconstruct
        reconstructed = multi_level_wavelet_reconstruction(dequantized_coeffs, sizes, decomposition_levels);
        
        % Adjust brightness
        reconstructed = reconstructed - min(reconstructed(:));
        reconstructed = reconstructed / max(reconstructed(:)) * 255;
        
        % Calculate PSNR
        psnr_values(i) = calculate_psnr(original_image, reconstructed);
    end
end

function compression_ratios = calculate_compression_ratios(original_image, quantization_steps, decomposition_levels)
    compression_ratios = zeros(size(quantization_steps));
    original_size_bits = numel(original_image) * 8;
    
    for i = 1:length(quantization_steps)
        step_size = quantization_steps(i);
        
        % Decomposition
        [coeffs, sizes] = multi_level_wavelet_decomposition(original_image, decomposition_levels);
        
        % Quantize
        quantized_coeffs = cell(size(coeffs));
        for j = 1:length(coeffs)
            quantized_coeffs{j} = round(coeffs{j} / step_size);
        end
        
        % Huffman encode
        total_compressed_bits = 0;
        for j = 1:length(quantized_coeffs)
            [encoded_data, ~] = huffman_encode(quantized_coeffs{j});
            total_compressed_bits = total_compressed_bits + length(encoded_data);
        end
        
        % Calculate compression ratio
        compression_ratios(i) = original_size_bits / total_compressed_bits;
    end
end

function save_color_compressed_data(filename, compressed_channels, huffman_tables, ...
    channel_sizes, decomposition_levels, step_size, image_size)
    
    % Save metadata
    metadata_filename = [filename(1:end-4) '_meta.mat'];
    save(metadata_filename, 'huffman_tables', 'channel_sizes', 'decomposition_levels', ...
        'step_size', 'image_size', '-v7.3');
    
    % Save encoded binary data
    fid = fopen(filename, 'wb');
    
    % Write number of channels
    num_channels = length(compressed_channels);
    fwrite(fid, num_channels, 'uint32');
    
    % For each channel
    for channel = 1:num_channels
        encoded_data = compressed_channels{channel};
        
        % Write number of subbands
        fwrite(fid, length(encoded_data), 'uint32');
        
        % Write each subband's encoded data
        for i = 1:length(encoded_data)
            % Store the original length
            data_length = length(encoded_data{i});
            fwrite(fid, data_length, 'uint32');
            
            % Write the actual data
            fwrite(fid, encoded_data{i}, 'uint8');
        end
    end
    
    fclose(fid);
end

% Add new function to load color compressed data
function [compressed_channels, huffman_tables, channel_sizes, decomposition_levels, ...
    step_size, image_size] = load_color_compressed_data(filename)
    
    % Load metadata
    metadata_filename = [filename(1:end-4) '_meta.mat'];
    metadata = load(metadata_filename);
    
    huffman_tables = metadata.huffman_tables;
    channel_sizes = metadata.channel_sizes;
    decomposition_levels = metadata.decomposition_levels;
    step_size = metadata.step_size;
    image_size = metadata.image_size;
    
    % Load binary data
    fid = fopen(filename, 'rb');
    
    % Read number of channels
    num_channels = fread(fid, 1, 'uint32');
    compressed_channels = cell(1, num_channels);
    
    % For each channel
    for channel = 1:num_channels
        % Read number of subbands
        num_subbands = fread(fid, 1, 'uint32');
        encoded_data = cell(1, num_subbands);
        
        % Read each subband's data
        for i = 1:num_subbands
            data_length = fread(fid, 1, 'uint32');
            encoded_data{i} = fread(fid, data_length, 'uint8')';
        end
        
        compressed_channels{channel} = encoded_data;
    end
    
    fclose(fid);
end