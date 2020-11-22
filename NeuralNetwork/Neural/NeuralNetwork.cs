using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Neural
{
    class NeuralNetwork
    {
        public double learning_rate = 0.1f;
        int _input_nodes;
        int _hidden_nodes;
        int _output_nodes;

        Matrix weights_ih;
        Matrix weights_ho;
        Matrix bias_h;
        Matrix bias_o;

        public NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes)
        {
            _input_nodes = input_nodes;
            _hidden_nodes = hidden_nodes;
            _output_nodes = output_nodes;

            weights_ih = new Matrix(_hidden_nodes, _input_nodes);
            weights_ho = new Matrix(_output_nodes, _hidden_nodes);
            weights_ih.Randomize();
            weights_ho.Randomize();

            bias_h = new Matrix(_hidden_nodes, 1);
            bias_o = new Matrix(_output_nodes, 1);
            bias_h.Randomize();
            bias_o.Randomize();

        }

        public double[] FeedForward(double[] input_array)
        {
            //Generating Hidden Outputs
            Matrix input = Matrix.FromArray(input_array);
            var hidden = Matrix.Multiply(this.weights_ih, input);
            hidden.Add(this.bias_h);

            //Activation function
            hidden.ApplyFunc(Sigmoid);

            var output = Matrix.Multiply(this.weights_ho, hidden);
            output.Add(bias_o);
            output.ApplyFunc(Sigmoid);

            return output.ToArray();
        }

        private double Sigmoid(double x)
        {
            return 1.0f / (1.0f + (double)Math.Exp(-x));
        }

        private double dSigmoid(double y)
        {
            //return Sigmoid(x) * (1 - Sigmoid(x));
            return y * (1 - y);
        }

        public void Train(double[] inputs, double[] targets)
        {
            //Generating Hidden Outputs
            Matrix input = Matrix.FromArray(inputs);
            var hidden = Matrix.Multiply(this.weights_ih, input);
            hidden.Add(this.bias_h);

            //Activation function
            hidden.ApplyFunc(Sigmoid);

            var outputs = Matrix.Multiply(this.weights_ho, hidden);
            outputs.Add(bias_o);
            outputs.ApplyFunc(Sigmoid);

            Matrix targets_m = Matrix.FromArray(targets);


            //-------------------------------------------------------------
            //Calculate the error
            var output_errors = Matrix.Subtract(targets_m, outputs);

            //Calculate gradient
            var gradients = Matrix.ApplyFunc(outputs, dSigmoid);
            gradients = Matrix.Multiply(outputs, output_errors);
            gradients = Matrix.MultiplyScalar(gradients, this.learning_rate);

            //Calculate deltas
            var hidden_t = Matrix.Transpose(hidden);
            var weights_ho_deltas = Matrix.Multiply(gradients, hidden_t);

            // Adjust the weights by deltas
            this.weights_ho.Add(weights_ho_deltas);

            //adjust the bias by its deltas
            this.bias_o.Add(gradients);


            //-----------------------------------------------------------------------
            //Calculate the hidden layer errors
            var who_t = Matrix.Transpose(this.weights_ho);
            var hidden_errors = Matrix.Multiply(who_t, output_errors);

            //calculate hidden gradient
            var hidden_gradient = Matrix.ApplyFunc(hidden, dSigmoid);
            hidden_gradient = Matrix.Multiply(hidden_gradient, hidden_errors);
            hidden_gradient = Matrix.MultiplyScalar(hidden_gradient, this.learning_rate);

            //Calculate hidden deltas
            var inputs_t = Matrix.Transpose(input);
            var weight_ih_deltas = Matrix.Multiply(hidden_gradient, inputs_t);

            // Adjust the weights by deltas
            this.weights_ih.Add(weight_ih_deltas);

            //adjust the bias by its deltas
            this.bias_h.Add(hidden_gradient);
        }
    }
}
