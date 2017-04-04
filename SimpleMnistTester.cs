using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNTest1Console
{
    class SimpleMnistTester
    {
        SimpleNeuralNet nn = new SimpleNeuralNet(784, 1568, 10, 0.02f);

        Stopwatch sw = new Stopwatch();

        int _trainCount = 10000;
        int _testCount = 2000;
        int _ephoc = 5;

        public void Run()
        {
            sw.Start();

            while (true)
            {
                try
                {
                    Console.Write("TRAIN >> ");

                    _trainCount = Convert.ToInt32(Console.ReadLine());

                    break;
                }
                catch
                {
                    continue;
                }
            }

            Learn();
            Printer.Print();

            Console.Beep(5000,750);
            
            while (true)
            {
                Printer.Print();

                lock (Printer.RenderLock)
                {
                    Console.Write("TEST >> ");
                    string cmd = Console.ReadLine();
                    if (cmd == "exit")
                    {
                        Environment.Exit(0);
                        break;
                    }
                    else
                    {
                        try
                        {
                            _testCount = Convert.ToInt32(cmd);
                        }
                        catch
                        {
                            continue;
                        }
                    }
                }

                Test();
                Printer.Print();
            }

            Console.ReadLine();
        }

        int learnCount = 0;
        int lps = 0;
        long lastMs = 0;
        private void Learn()
        {
            string[] trainLines = File.ReadAllLines(System.IO.Path.Combine(Environment.CurrentDirectory, "mnist_train.csv"));

            int countMax = _trainCount;
            for (int i = 0; i < _ephoc; i++)
            {
                int lineInd = 0;

                string[] spl = new string[785];
                string[] data = new string[784];
                float[] learnArray = new float[data.Length];
                float[] targetArray = new float[10];
                
                foreach (string line in trainLines)
                {
                    line.Split(',').CopyTo(spl, 0);
                    Array.Copy(spl, 1, data, 0, data.Length);

                    int ind = 0;
                    foreach (string s in data)
                    {
                        learnArray[ind] = (float)Convert.ToInt32(s) / 255 * 0.99f + 0.01f;
                        ind++;
                    }

                    for (int ii = 0; ii < targetArray.Length; ii++)
                    {
                        targetArray[ii] = 0.01f;
                    }
                    targetArray[Convert.ToInt32(spl[0])] = 0.99f;
                    
                    nn.Train(learnArray, targetArray);

                    learnCount++;
                    if (sw.ElapsedMilliseconds - lastMs > 250)
                    {
                        lastMs = sw.ElapsedMilliseconds;
                        lps = learnCount * 4;
                        learnCount = 0;
                    }

                    lock (Printer.RenderLock)
                    {
                        Printer.Clear();

                        PrintMnist(learnArray);

                        PrintTrainStep(i, lineInd, Math.Min(countMax, trainLines.Length), spl[0], lps);
                    }

                    lineInd++;
                    if (lineInd >= countMax || lineInd >= trainLines.Length)
                        break;
                }
            }
        }

        private void Test()
        {
            string[] trainLines = File.ReadAllLines(System.IO.Path.Combine(Environment.CurrentDirectory, "mnist_test.csv"));

            int countMax = _testCount;
            int lineInd = 0;

            int testCount = 0;
            int testSum = 0;

            foreach (string line in trainLines)
            {
                lock (Printer.RenderLock)
                {
                    Printer.Clear();

                    string[] spl = line.Split(',');
                    string[] data = new string[spl.Length - 1];
                    Array.Copy(spl, 1, data, 0, data.Length);

                    float[] queryArray = new float[data.Length];
                    int ind = 0;
                    foreach (string s in data)
                    {
                        queryArray[ind] = (float)Convert.ToInt32(s) / 255 * 0.99f + 0.01f;
                        ind++;
                    }

                    PrintMnist(queryArray);

                    int answer = nn.Query(queryArray);
                    if (answer == Convert.ToInt32(spl[0]))
                    {
                        testSum++;
                    }
                    testCount++;
                    
                    PrintTestStep(lineInd, Math.Min(countMax, trainLines.Length), spl[0], (float)testSum / testCount);

                    lineInd++;
                    if (lineInd >= countMax || lineInd >= trainLines.Length)
                        break;
                }
            }
        }

        private void PrintTrainStep(int epoch, int count, int maxcount, string answer, int lps)
        {
            string str = string.Format("[EPOCH:{0}] [TRAIN:\"{1}\"] {2} {3} ({4}) LearnPerSecond: {5}", epoch + 1, answer, StrLen(count.ToString() + "/" + maxcount.ToString(), 15), StrLen((((double)count / maxcount) * 100).ToString("0.00") + "%", 6),
                StrLen(StrMul("=", (int)Math.Round(((double)count / maxcount) * 50)), 50, "-"), lps);

            Printer.WriteLine(str);
        }

        private void PrintTestStep(int count, int maxcount, string answer, float occuracy)
        {
            string str = string.Format("(Occuracy: {0}) [TEST:\"{1}\"] {2} {3} ({4})", occuracy.ToString("0.00000"), answer, StrLen(count.ToString() + "/" + maxcount.ToString(), 15), StrLen((((double)count / maxcount) * 100).ToString("0.00") + "%", 6),
                StrLen(StrMul("=", (int)Math.Round(((double)count / maxcount) * 50)), 50, "-"));

            Printer.WriteLine(str);
        }

        StringBuilder StringMultiplyBuilder = new StringBuilder();
        private string StrMul(string str, int count)
        {
            StringMultiplyBuilder.Clear();
            StringMultiplyBuilder.Append(str);

            string o = str;

            for (int i = 0; i < count - 1; i++)
            {
                StringMultiplyBuilder.Append(o);
            }

            return StringMultiplyBuilder.ToString();
        }

        StringBuilder StringLengthBuilder = new StringBuilder();
        private string StrLen(string str, int count, string whitespace = " ")
        {
            StringLengthBuilder.Clear();
            StringLengthBuilder.Append(str);

            int c = count - str.Length;
            for (int i = 0; i < c; i++)
            {
                StringLengthBuilder.Append(whitespace);
            }
            return StringLengthBuilder.ToString();
        }

        StringBuilder MnistBuilder = new StringBuilder();

        private void PrintMnist(float[] inp)
        {
            MnistBuilder.Clear();

            for (int column = 0; column < 28; column++)
            {
                for (int row = 0; row < 28; row++)
                {
                    MnistBuilder.Append(GetAscii(inp[column * 28 + row]));
                }

                MnistBuilder.Append("\n");
            }

            Printer.WriteLine(MnistBuilder.ToString());
        }

        private char[] asciiRecord = { ' ', '.', ',', '*', ':', '|', 'I', '&', '%', '#', '█' };
        private char GetAscii(float f)
        {
            int ind = (int)Math.Ceiling(f * (asciiRecord.Length - 1));

            return asciiRecord[ind];
        }
    }
}
