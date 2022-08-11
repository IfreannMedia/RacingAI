using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ANNDrive : MonoBehaviour
{

    ANN ann;
    public float visibleDistance = 200;
    public int epochs = 1000;
    public float speed = 50.0f;
    public float rotationSpeed = 100.0f;

    bool trainingDone = false;
    float trainingProgress = 0;
    // sum of squared errors and last SSE
    double sse = 0;
    double lastSSE = 1;

    public float translation;
    public float rotation;

    void Start()
    {
        ann = new ANN(5, 2, 1, 10, 0.5);
        StartCoroutine(LoadTrainingSet());
    }

    private void OnGUI()
    {
        GUI.Label(new Rect(22, 25, 250, 30), "SSE: " + lastSSE);
        GUI.Label(new Rect(22, 40, 250, 30), "ALPHA: " + ann.alpha);
        GUI.Label(new Rect(22, 55, 250, 30), "TRAINED: " + trainingProgress);
    }

    IEnumerator LoadTrainingSet()
    {
        string path = Application.dataPath + "/trainingData.txt";
        string line;
        if (File.Exists(path))
        {
            int lineCount = File.ReadAllLines(path).Length;
            // training data file
            StreamReader tdf = File.OpenText(path);
            // calcOutputs - what the NN is calculating to send back
            List<double> calcOutputs = new List<double>();
            List<double> inputs = new List<double>();
            List<double> outputs = new List<double>();

            for (int i = 0; i < epochs; i++)
            {
                // set errors to 0 and go back to start of training file
                sse = 0;
                tdf.BaseStream.Position = 0;

                while ((line = tdf.ReadLine()) != null)
                {
                    // comma seperated training data into array
                    string[] data = line.Split(",");
                    float thisError = 0;
                    // ignore any training data = 0
                    if (System.Convert.ToDouble(data[5]) != 0
                        && System.Convert.ToDouble(data[6]) != 0)
                    {
                        inputs.Clear();
                        outputs.Clear();
                        inputs.Add(System.Convert.ToDouble(data[0]));
                        inputs.Add(System.Convert.ToDouble(data[1]));
                        inputs.Add(System.Convert.ToDouble(data[2]));
                        inputs.Add(System.Convert.ToDouble(data[3]));
                        inputs.Add(System.Convert.ToDouble(data[4]));

                        double o1 = Map(0, 1, -1, 1, System.Convert.ToSingle(data[5]));
                        outputs.Add(o1);
                        double o2 = Map(0, 1, -1, 1, System.Convert.ToSingle(data[6]));
                        outputs.Add(o2);

                        // get the outputs one for translation and one for rotation
                        calcOutputs = ann.Train(inputs, outputs);
                        // sum of squares errors on both outputs, averaged
                        thisError = ((Mathf.Pow((float)(outputs[0] - calcOutputs[0]), 2) +
                            Mathf.Pow((float)(outputs[1] - calcOutputs[1]), 2))) / 2.0f;
                    }
                    sse += thisError;
                }
                trainingProgress = (float)i / (float)epochs;
                sse /= lineCount;
                lastSSE = sse;

                yield return null;
            }
        }
        trainingDone = true;
    }

    float Map(float newFrom, float newTo, float origFrom, float origTo, float value)
    {
        if (value <= origFrom)
            return newFrom;
        else if (value >= origTo)
            return newTo;
        return (newTo - newFrom) * ((value - origFrom) / (origTo - origFrom)) + newFrom;
    }

    float Round(float x)
    {
        return (float)System.Math.Round(x, System.MidpointRounding.AwayFromZero) / 2.0f;
    }

    private void Update()
    {
        /*
        //left 45
        Quaternion.AngleAxis(45, Vector3.up) * -this.transform.right;

        //right 45
        Quaternion.AngleAxis(-45, Vector3.up) * this.transform.right;
         */
    }

}
