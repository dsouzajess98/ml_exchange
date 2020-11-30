pragma solidity ^0.4.11;

contract ml_exchange_test {

        // Fallback function for sending ether to this contract
        function () public payable {}

        function ml_exchange_test() {
            init_level[0] = 10;
        }

        struct Submission {
          address payment_address;
          uint num_neurons_input_layer;
          uint num_neurons_output_layer;
          uint[] num_neurons_hidden_layer;
          int256[] weights;
          int256[] biases;
        }
          struct NeuralLayer {
            int256[] neurons;
            int256[] errors;
            string layer_type;
          }

        uint constant no_of_contracts = 30;
        address[no_of_contracts] public organizer;
        uint[no_of_contracts] public best_submission_index;
        int256[no_of_contracts] public best_submission_accuracy;
        int256[no_of_contracts] public model_accuracy_criteria;
        bool[no_of_contracts] public use_test_data;
        uint constant partition_size = 25;
        uint constant datapoint_size = 3;
        uint constant prediction_size = 1;
        uint16 constant max_num_data_groups = 500;
        uint16 constant training_data_group_size = 400;
        uint16 constant testing_data_group_size = max_num_data_groups - training_data_group_size;
        bytes32[max_num_data_groups/partition_size][no_of_contracts] hashed_data_groups;
        uint[max_num_data_groups/partition_size][no_of_contracts] data_group_nonces;

        int256[datapoint_size][][no_of_contracts] public train_data;
        int256[datapoint_size][][no_of_contracts] public test_data;
        bytes32 partition_seed;

        uint[no_of_contracts] public submission_stage_block_size;
        uint[no_of_contracts] public evaluation_stage_block_size;
        uint[no_of_contracts] public reveal_test_data_groups_block_size;

        uint[no_of_contracts] public init1_block_height;
        uint[no_of_contracts] public init3_block_height;
        uint[no_of_contracts] public init_level;
        uint[training_data_group_size/partition_size][no_of_contracts] public training_partition;
        uint[testing_data_group_size/partition_size][no_of_contracts] public testing_partition;
        uint256[no_of_contracts] public train_dg_revealed;
        uint256[no_of_contracts] public test_dg_revealed;
        Submission[][no_of_contracts] submission_queue;
        bool[no_of_contracts] public contract_terminated;
        int constant int_precision = 10000;

    function tester(uint contract_id) public payable{
        init_level[0] = 14;
    }

    function init1(bytes32[max_num_data_groups/partition_size] _hashed_data_groups, int accuracy_criteria, address organizer_refund_address, uint submission_t, uint contract_id) public payable {

        contract_terminated[contract_id] = false;
        submission_stage_block_size[contract_id] = submission_t;
        evaluation_stage_block_size[contract_id] = 1;
        reveal_test_data_groups_block_size[contract_id] = 4;
        init_level[contract_id] = 0;
        use_test_data[contract_id] = false;
        test_dg_revealed[contract_id] = 0;
        train_dg_revealed[contract_id] = 0;
        organizer[contract_id] = organizer_refund_address;
        init_level[contract_id] = 1;
        init1_block_height[contract_id] = block.number;
        assert(_hashed_data_groups.length == max_num_data_groups/partition_size);
        hashed_data_groups[contract_id] = _hashed_data_groups;
        assert(accuracy_criteria > 0);
        model_accuracy_criteria[contract_id] = accuracy_criteria;
  }

    function init2(uint contract_id) public payable {

        assert(contract_terminated[contract_id] == false);
        // Only allow calling it once, in order
        assert(init_level[contract_id] == 1);
        uint[] memory index_array = new uint[](max_num_data_groups/partition_size);
          for (uint i = 0; i < max_num_data_groups/partition_size; i++)
          {
            index_array[i] = i;
          }
          randomly_select_index(index_array, contract_id);
        init_level[contract_id]=2;

  }

  function init3(int256[] _train_data_groups, int256 _train_data_group_nonces, uint contract_id) external {
    assert(contract_terminated[contract_id] == false);
    assert(init_level[contract_id] == 2);
    assert((_train_data_groups.length/partition_size)/datapoint_size == 1);
    assert(sha_data_group(_train_data_groups, _train_data_group_nonces) ==
      hashed_data_groups[contract_id][training_partition[contract_id][train_dg_revealed[contract_id]]]);
    train_dg_revealed[contract_id] += 1;
    unpack_data_groups(_train_data_groups, true, contract_id);
    if (train_dg_revealed[contract_id] == (training_data_group_size/partition_size)) {
      init_level[contract_id] = 3;
      init3_block_height[contract_id] = block.number;
    }
  }

  function get_training_index(uint contract_id) public view returns(uint[training_data_group_size/partition_size]) {
    return training_partition[contract_id];
  }

  function get_testing_index(uint contract_id) public view returns(uint[testing_data_group_size/partition_size]) {
    return testing_partition[contract_id];
  }

  function get_submission_queue_length(uint contract_id) public view returns(uint) {
    return submission_queue[contract_id].length;
  }

  function submit_model(
    // Public function for users to submit a solution
    address payment_address,
    uint num_neurons_input_layer,
    uint num_neurons_output_layer,
    uint[] num_neurons_hidden_layer,
    int[] weights,
    int256[] biases,
    uint contract_id) public {
      // Make sure contract is not terminated
      assert(contract_terminated[contract_id] == false);
      // Make sure it's not the initialization stage anymore
      assert(init_level[contract_id] == 3);
      // Make sure that num of neurons in the input & output layer matches
      // the problem description
      assert(num_neurons_input_layer == datapoint_size - prediction_size);
      // Because we can encode binary output in two different ways, we check
      // for both of them
      assert(num_neurons_output_layer == prediction_size || num_neurons_output_layer == (prediction_size+1));
      // Make sure that the number of weights match network structure
      assert(valid_weights(weights, num_neurons_input_layer, num_neurons_output_layer, num_neurons_hidden_layer));
      // Add solution to submission queue
      Submission[] x;
      x.push(Submission(
        payment_address,
        num_neurons_input_layer,
        num_neurons_output_layer,
        num_neurons_hidden_layer,
        weights,
        biases));
      submission_queue[contract_id] = x;
  }

  function get_submission_id(
    // Public function that returns the submission index ID
    address paymentAddress,
    uint num_neurons_input_layer,
    uint num_neurons_output_layer,
    uint[] num_neurons_hidden_layer,
    int[] weights,
    int256[] biases,
    uint contract_id) public view returns (uint) {
      // Iterate over submission queue to get submission index ID
      for (uint i = 0; i < submission_queue[contract_id].length; i++) {
        if (submission_queue[contract_id][i].payment_address != paymentAddress) {
          continue;
        }
        if (submission_queue[contract_id][i].num_neurons_input_layer != num_neurons_input_layer) {
          continue;
        }
        if (submission_queue[contract_id][i].num_neurons_output_layer != num_neurons_output_layer) {
          continue;
        }
        for (uint j = 0; j < num_neurons_hidden_layer.length; j++) {
            if (submission_queue[contract_id][i].num_neurons_hidden_layer[j] != num_neurons_hidden_layer[j]) {
              continue;
            }
        }
        for (uint k = 0; k < weights.length; k++) {
            if (submission_queue[contract_id][i].weights[k] != weights[k]) {
              continue;
            }
        }
        for (uint l = 0; l < biases.length; l++) {
          if (submission_queue[contract_id][i].biases[l] != biases[l]) {
            continue;
          }
        }
        // If everything matches, return the submission index
        return i;
      }
      // If submission is not in the queue, just throw an exception
      require(false);
  }

    function reveal_test_data(int256[] _test_data_groups, int256 _test_data_group_nonces, uint contract_id) external {
    // Make sure contract is not terminated
    //assert(contract_terminated[contract_id] == false);
    // Make sure it's not the initialization stage anymore
    //assert(init_level[contract_id] == 3);
    // Verify data group and nonce lengths
    //assert((_test_data_groups.length/partition_size)/datapoint_size == 1);
    // Verify data group hashes
    //assert(sha_data_group(_test_data_groups, _test_data_group_nonces) == hashed_data_groups[contract_id][testing_partition[contract_id][test_dg_revealed]]);
    test_dg_revealed[contract_id] += 1;
    // Assign testing data after verifying the corresponding hash
    unpack_data_groups(_test_data_groups, false, contract_id);
    // Use test data for evaluation
    use_test_data[contract_id] = true;
  }

  function evaluate_model(uint submission_index, uint contract_id, int256 score) public {
    // Make sure contract is not terminated
    //assert(contract_terminated[contract_id] == false);
    // Make sure it's not the initialization stage anymore
    //assert(init_level[contract_id] == 3);
    // Evaluates a submitted model & keeps track of the best model
    int256 submission_accuracy = 0;
    //if (use_test_data[contract_id] == true) {
     // submission_accuracy = model_accuracy(submission_index, test_data[contract_id], contract_id);
   // } else {
   //   submission_accuracy = model_accuracy(submission_index, train_data[contract_id], contract_id);
   // }
   submission_accuracy = score;

    // Keep track of the most accurate model
    if (submission_accuracy > best_submission_accuracy[contract_id]) {
      best_submission_index[contract_id] = submission_index;
      best_submission_accuracy[contract_id] = submission_accuracy;
    }
    // If accuracy is the same, the earlier submission is selected
    if (submission_accuracy == best_submission_accuracy[contract_id]) {
      if (submission_index < best_submission_index[contract_id]) {
        best_submission_index[contract_id] = submission_index;
        best_submission_accuracy[contract_id] = submission_accuracy;
      }
    }
  }

  function cancel_contract(uint contract_id) public {
    // Make sure contract is not already terminated
    assert(contract_terminated[contract_id] == false);
    // Contract can only be cancelled if initialization has failed.
    assert(init_level[contract_id] < 3);
    // Refund remaining balance to organizer
    organizer[contract_id].transfer(this.balance);
    // Terminate contract
    contract_terminated[contract_id] = true;
  }

  function finalize_contract(uint contract_id) public {
    // Make sure contract is not terminated
    //assert(contract_terminated[contract_id] == false);
    // Make sure it's not the initialization stage anymore
    //assert(init_level[contract_id] == 3);
    // Get the best submission to compare it against the criteria
    // Submission memory best_submission = submission_queue[contract_id][best_submission_index[contract_id]];
    // If best submission passes criteria, payout to the submitter
    // if (best_submission_accuracy[contract_id] >= model_accuracy_criteria[contract_id]) {
     // best_submission.payment_address.transfer(this.balance);
    // If the best submission fails the criteria, refund the balance back to the organizer
   // } else {
    //  organizer[contract_id].transfer(this.balance);
    // }
    contract_terminated[contract_id] = true;
  }

  function model_accuracy(uint submission_index, int256[datapoint_size][] data, uint contract_id) public constant returns (int256){
    // Make sure contract is not terminated
    assert(contract_terminated[contract_id] == false);
    // Make sure it's not the initialization stage anymore
    assert(init_level[contract_id] == 3);
    // Leave function public for offline error calculation
    // Get's the sum error for the model
    Submission memory sub = submission_queue[contract_id][submission_index];
    int256 true_prediction = 0;
    int256 false_prediction = 0;
    bool one_hot; // one-hot encoding if prediction size is 1 but model output size is 2
    int[] memory prediction;
    int[] memory ground_truth;
    if ((prediction_size + 1) == sub.num_neurons_output_layer) {
      one_hot = true;
      prediction = new int[](sub.num_neurons_output_layer);
      ground_truth = new int[](sub.num_neurons_output_layer);
    } else {
      one_hot = false;
      prediction = new int[](prediction_size);
      ground_truth = new int[](prediction_size);
    }
    for (uint i = 0; i < data.length; i++) {
      // Get ground truth
      for (uint j = datapoint_size-prediction_size; j < data[i].length; j++) {
        uint d_index = j - datapoint_size + prediction_size;
        // Only get prediction values
        if (one_hot == true) {
          if (data[i][j] == 0) {
            ground_truth[d_index] = 1;
            ground_truth[d_index + 1] = 0;
          } else if (data[i][j] == 1) {
            ground_truth[d_index] = 0;
            ground_truth[d_index + 1] = 1;
          } else {
            // One-hot encoding for more than 2 classes is not supported
            require(false);
          }
        } else {
          ground_truth[d_index] = data[i][j];
        }
      }
      // Get prediction
      prediction = get_prediction(sub, data[i]);
      // Get error for the output layer
      for (uint k = 0; k < ground_truth.length; k++) {
        if (ground_truth[k] == prediction[k]) {
          true_prediction += 1;
        } else {
          false_prediction += 1;
        }
      }
    }
    // We multipl by int_precision to get up to x decimal point precision while
    // calculating the accuracy
    return (true_prediction * int_precision) / (true_prediction + false_prediction);
  }

  function get_train_data_length(uint contract_id) public view returns(uint256) {
    return train_data[contract_id].length;
  }

  function get_test_data_length(uint contract_id) public view returns(uint256) {
    return test_data[contract_id].length;
  }

  function round_up_division(int256 dividend, int256 divisor) private pure returns(int256) {
    // A special trick since solidity normall rounds it down
    return (dividend + divisor -1) / divisor;
  }

  function not_in_train_partition(uint[training_data_group_size/partition_size] partition, uint number) private pure returns (bool) {
    for (uint i = 0; i < partition.length; i++) {
      if (number == partition[i]) {
        return false;
      }
    }
    return true;
  }

  function randomly_select_index(uint[] array, uint contract_id) private {
    uint t_index = 0;
    uint array_length = array.length;
    uint block_i = 0;
    // Randomly select training indexes
    while(t_index < training_partition[contract_id].length) {
      uint random_index = uint(sha256(block.blockhash(block.number-block_i))) % array_length;
      training_partition[contract_id][t_index] = array[random_index];
      array[random_index] = array[array_length-1];
      array_length--;
      block_i++;
      t_index++;
    }
    t_index = 0;
    while(t_index < testing_partition[contract_id].length) {
      testing_partition[contract_id][t_index] = array[array_length-1];
      array_length--;
      t_index++;
    }


  }

  function valid_weights(int[] weights, uint num_neurons_input_layer, uint num_neurons_output_layer, uint[] num_neurons_hidden_layer) private pure returns (bool) {
    // make sure the number of weights match the network structure
    // get number of weights based on network structure
    uint ns_total = 0;
    uint wa_total = 0;
    uint number_of_layers = 2 + num_neurons_hidden_layer.length;

    if (number_of_layers == 2) {
      ns_total = num_neurons_input_layer * num_neurons_output_layer;
    } else {
      for(uint i = 0; i < num_neurons_hidden_layer.length; i++) {
        // Get weights between first hidden layer and input layer
        if (i==0){
          ns_total += num_neurons_input_layer * num_neurons_hidden_layer[i];
        // Get weights between hidden layers
        } else {
          ns_total += num_neurons_hidden_layer[i-1] * num_neurons_hidden_layer[i];
        }
      }
      // Get weights between last hidden layer and output layer
      ns_total += num_neurons_hidden_layer[num_neurons_hidden_layer.length-1] * num_neurons_output_layer;
    }
    // get number of weights in the weights array
    wa_total = weights.length;

    return ns_total == wa_total;
  }

    function unpack_data_groups(int256[] _data_groups, bool is_train_data, uint contract_id) private {
    int256[datapoint_size][] memory merged_data_group = new int256[datapoint_size][](_data_groups.length/datapoint_size);

    for (uint i = 0; i < _data_groups.length/datapoint_size; i++) {
      for (uint j = 0; j < datapoint_size; j++) {
        merged_data_group[i][j] = _data_groups[i*datapoint_size + j];
      }
    }
    if (is_train_data == true) {
      // Assign training data
      for (uint k = 0; k < merged_data_group.length; k++) {
        train_data[contract_id].push(merged_data_group[k]);
      }
    } else {
      // Assign testing data
      for (uint l = 0; l < merged_data_group.length; l++) {
        test_data[contract_id].push(merged_data_group[l]);
      }
    }
  }

    function sha_data_group(int256[] data_group, int256 data_group_nonce) private pure returns (bytes32) {
      // Extract the relevant data points for the given data group index
      // We concat all data groups and add the nounce to the end of the array
      // and get the sha256 for the array
      uint index_tracker = 0;
      uint256 total_size = datapoint_size * partition_size;
      /* uint256 start_index = data_group_index * total_size;
      uint256 iter_limit = start_index + total_size; */
      int256[] memory all_data_points = new int256[](total_size+1);

      for (uint256 i = 0; i < total_size; i++) {
        all_data_points[index_tracker] = data_group[i];
        index_tracker += 1;
      }
      // Add nonce to the whole array
      all_data_points[index_tracker] = data_group_nonce;
      // Return sha256 on all data points + nonce
      return sha256(all_data_points);
    }

  function relu_activation(int256 x) private pure returns (int256) {
    if (x < 0) {
      return 0;
    } else {
      return x;
    }
  }

  function get_layer(uint nn) private pure returns (int256[]) {
    int256[] memory input_layer = new int256[](nn);
    return input_layer;
  }

  function get_hidden_layers(uint[] l_nn) private pure returns (int256[]) {
    uint total_nn = 0;
    // Skip first and last layer since they're not hidden layers
    for (uint i = 1; i < l_nn.length-1; i++) {
      total_nn += l_nn[i];
    }
    int256[] memory hidden_layers = new int256[](total_nn);
    return hidden_layers;
  }

  function access_hidden_layer(int256[] hls, uint[] l_nn, uint index) private pure returns (int256[]) {
    // Returns the hidden layer from the hidden layers array
    int256[] memory hidden_layer = new int256[](l_nn[index+1]);
    uint hidden_layer_index = 0;
    uint start = 0;
    uint end = 0;
    for (uint i = 0; i < index; i++) {
      start += l_nn[i+1];
    }
    for (uint j = 0; j < (index + 1); j++) {
      end += l_nn[j+1];
    }
    for (uint h_i = start; h_i < end; h_i++) {
      hidden_layer[hidden_layer_index] = hls[h_i];
      hidden_layer_index += 1;
    }
    return hidden_layer;
  }

  function get_prediction(Submission sub, int[datapoint_size] data_point) private pure returns(int256[]) {
    uint[] memory l_nn = new uint[](sub.num_neurons_hidden_layer.length + 2);
    l_nn[0] = sub.num_neurons_input_layer;
    for (uint i = 0; i < sub.num_neurons_hidden_layer.length; i++) {
      l_nn[i+1] = sub.num_neurons_hidden_layer[i];
    }
    l_nn[sub.num_neurons_hidden_layer.length+1] = sub.num_neurons_output_layer;
    return forward_pass(data_point, sub.weights, sub.biases, l_nn);
  }

  function forward_pass(int[datapoint_size] data_point, int256[] weights, int256[] biases, uint[] l_nn) private pure returns (int256[]) {
    // Initialize neuron arrays
    int256[] memory input_layer = get_layer(l_nn[0]);
    int256[] memory hidden_layers = get_hidden_layers(l_nn);
    int256[] memory output_layer = get_layer(l_nn[l_nn.length-1]);

    // load inputs from input layer
    for (uint input_i = 0; input_i < l_nn[0]; input_i++) {
      input_layer[input_i] = data_point[input_i];
    }
    return forward_pass2(l_nn, input_layer, hidden_layers, output_layer, weights, biases);
  }

  function forward_pass2(uint[] l_nn, int256[] input_layer, int256[] hidden_layers, int256[] output_layer, int256[] weights, int256[] biases) public pure returns (int256[]) {
    // index_counter[0] is weight index
    // index_counter[1] is hidden_layer_index
    uint[] memory index_counter = new uint[](2);
    for (uint layer_i = 0; layer_i < (l_nn.length-1); layer_i++) {
      int256[] memory current_layer;
      int256[] memory prev_layer;
      // If between input and first hidden layer
      if (hidden_layers.length != 0) {
        if (layer_i == 0) {
          current_layer = access_hidden_layer(hidden_layers, l_nn, layer_i);
          prev_layer = input_layer;
        // If between output and last hidden layer
        } else if (layer_i == (l_nn.length-2)) {
          current_layer = output_layer;
          prev_layer = access_hidden_layer(hidden_layers, l_nn, (layer_i-1));
        // If between hidden layers
        } else {
          current_layer = access_hidden_layer(hidden_layers, l_nn, layer_i);
          prev_layer = access_hidden_layer(hidden_layers, l_nn, layer_i-1);
        }
      } else {
        current_layer = output_layer;
        prev_layer = input_layer;
      }
      for (uint layer_neuron_i = 0; layer_neuron_i < current_layer.length; layer_neuron_i++) {
        int total = 0;
        for (uint prev_layer_neuron_i = 0; prev_layer_neuron_i < prev_layer.length; prev_layer_neuron_i++) {
          total += prev_layer[prev_layer_neuron_i] * weights[index_counter[0]];
          index_counter[0]++;
        }
        total += biases[layer_i];
        total = total / int_precision; // Divide by int_precision to scale down
        // If between output and last hidden layer
        if (layer_i == (l_nn.length-2)) {
            output_layer[layer_neuron_i] = relu_activation(total);
        } else {
            hidden_layers[index_counter[1]] = relu_activation(total);
        }
        index_counter[1]++;
      }
    }
    return output_layer;
  }
}