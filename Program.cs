using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime;
using System.Text;
using System.Threading.Tasks;

namespace NNTest1Console
{
    class Program
    {
        static void Main(string[] args)
        {
            GCSettings.LatencyMode = GCLatencyMode.Batch;

            SimpleMnistTester tester = new SimpleMnistTester();

            tester.Run();

            Console.ReadLine();
        }
    }
}
