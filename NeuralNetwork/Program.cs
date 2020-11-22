using NeuralNetwork.Neural;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var training_data = new List<Data>();

            training_data.Add(new Data(new double[] { 0f, 0f }, new double[] { 0}));
            training_data.Add(new Data(new double[] { 1, 0f }, new double[] { 1 }));
            training_data.Add(new Data(new double[] { 0, 1 }, new double[] { 1 }));
            training_data.Add(new Data(new double[] { 1, 1 }, new double[] { 0}));

            var nn = new Neural.NeuralNetwork(2, 15, 1);

            while (true)
            {
                Console.WriteLine("What do you want me to do?");
                string press = Console.ReadLine();

                if (press == "@train")
                {
                    Console.Clear();
                    Console.WriteLine("training...");
                    for (var i = 0; i < 10000; i++)
                    {
                        List<Data> tmp = new List<Data>(training_data);

                        do
                        {
                            var index = new Random().Next(tmp.Count);
                            var data = tmp[index];
                            tmp.RemoveAt(index);
                            nn.Train(data.inputs, data.targets);
                        } while (tmp.Count > 0);
                    }
                    Console.Clear();
                    Console.WriteLine("trained!");
                }
                else if (press == "@eval")
                {
                    Console.Clear();
                    LogResult(nn.FeedForward(new double[] { 1, 1 }), "0");
                    LogResult(nn.FeedForward(new double[] { 0, 1 }), "1");
                }
            }
        }

        static void LogResult(double[] data, string expected)
        {
            Console.Write("Should be: " + expected + "-> ");
            for (var i = 0; i < data.Length; i++)
            {
                Console.Write(Math.Round(data[i], 2) + "; ");
            }
            Console.WriteLine();
        }

        public class Data
        {
            public double[] inputs { get; set; }
            public double[] targets { get; set; }

            public Data(double[] i, double[] t)
            {
                inputs = i;
                targets = t;
            }
        }
    }
}
