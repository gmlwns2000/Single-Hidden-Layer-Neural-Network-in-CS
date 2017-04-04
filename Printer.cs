using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace NNTest1Console
{
    public static class Printer
    {
        private static StringBuilder builder = new StringBuilder();

        public static object RenderLock = new object();

        public static int FPS = 24;

        static Printer()
        {
            Timer t = new Timer();
            t.Interval = (double)1000 / 24;
            t.Elapsed += T_Elapsed;
            t.Start();
        }

        public static void Clear()
        {
            builder.Clear();

            Console.CursorLeft = 0;
            Console.CursorTop = 0;
        }

        public static void WriteLine(string text)
        {
            builder.AppendLine(text);
        }

        public static void Print()
        {
            lock (RenderLock)
            {
                Console.Clear();

                Console.CursorLeft = 0;
                Console.CursorTop = 0;

                Console.Write(builder);
            }
        }

        private static void T_Elapsed(object sender, ElapsedEventArgs e)
        {
            lock (RenderLock)
            {
                Console.CursorLeft = 0;
                Console.CursorTop = 0;

                Console.Write(builder);
            }
        }
    }
}
