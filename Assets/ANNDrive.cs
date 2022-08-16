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

    public bool loadFromFile;

    void Start()
    {
        ann = new ANN(5, 2, 1, 10, 0.5);
        if (loadFromFile)
        {
            LoadWeightsFromFile();
            trainingDone = true;
        }
        else
        {
            StartCoroutine(LoadTrainingSet());
        }
    }

    private void SaveWeightsToFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine(ann.PrintWeights());
        wf.Close();
    }

    private void LoadWeightsFromFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamReader wf = File.OpenText(path);
        if(File.Exists(path))
        {
            string line = wf.ReadLine();
            ann.LoadWeights(line);
        }

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
    void Update() {
        if(!trainingDone) return;

        List<double> calcOutputs = new List<double>();
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        //raycasts
        RaycastHit hit;
        float fDist = 0, rDist = 0, lDist = 0, r45Dist = 0, l45Dist = 0;

        //forward
        if (Physics.Raycast(transform.position, this.transform.forward, out hit, visibleDistance))
        {
            fDist = 1-Round(hit.distance/visibleDistance);
        }

        //right
        if (Physics.Raycast(transform.position, this.transform.right, out hit, visibleDistance))
        {
            rDist = 1-Round(hit.distance/visibleDistance);
        }

        //left
        if (Physics.Raycast(transform.position, -this.transform.right, out hit, visibleDistance))
        {
            lDist = 1-Round(hit.distance/visibleDistance);
        }

        //right 45
        if (Physics.Raycast(transform.position, 
                            Quaternion.AngleAxis(-45, Vector3.up) * this.transform.right, out hit, visibleDistance))
        {
            r45Dist = 1-Round(hit.distance/visibleDistance);
        }

        //left 45
        if (Physics.Raycast(transform.position, 
                            Quaternion.AngleAxis(45, Vector3.up) * -this.transform.right, out hit, visibleDistance))
        {
            l45Dist = 1-Round(hit.distance/visibleDistance);
        }

        inputs.Add(fDist);
        inputs.Add(rDist);
        inputs.Add(lDist);
        inputs.Add(r45Dist);
        inputs.Add(l45Dist);
        outputs.Add(0);
        outputs.Add(0);
        calcOutputs = ann.CalcOutput(inputs,outputs);
        float translationInput = Map(-1,1,0,1,(float) calcOutputs[0]);
        float rotationInput = Map(-1,1,0,1,(float) calcOutputs[1]);
        translation = translationInput * speed * Time.deltaTime;
        rotation = rotationInput * rotationSpeed * Time.deltaTime;
        this.transform.Translate(0, 0, translation);
        this.transform.Rotate(0, rotation, 0);        

    }
}
