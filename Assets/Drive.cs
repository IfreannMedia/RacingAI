using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class Drive : MonoBehaviour
{
    public float speed = 90.0f;
    public float rotationSpeed = 150;
    public float visibleDistance = 200;
    List<string> collectedTrainingData = new List<string>();
    StreamWriter sw;
    private void Start()
    {
        string path = Application.dataPath + "/trainingData.txt";
        sw = File.CreateText(path);
    }

    private void OnApplicationQuit()
    {
        foreach (string tf in collectedTrainingData)
        {
            sw.WriteLine(tf);
        }
        sw.Close();
    }

    float Round(float x)
    {
        return (float)System.Math.Round(x, System.MidpointRounding.AwayFromZero) / 2.0f;
    }

    void Update()
    {
        // Get the horizontal and vertical axis.
        // By default they are mapped to the arrow keys.
        // The value is in the range -1 to 1
        float translationInput = Input.GetAxis("Vertical");
        float rotationInput = Input.GetAxis("Horizontal");

        // Make it move 10 meters per second instead of 10 meters per frame...
        float translation = Time.deltaTime * speed * translationInput;
        float rotation = Time.deltaTime * rotationSpeed * rotationInput;

        // Move translation along the object's z-axis
        transform.Translate(0, 0, translation);

        // Rotate around our y-axis
        transform.Rotate(0, rotation, 0);

        Debug.DrawRay(transform.position, transform.forward * visibleDistance, Color.red);
        Debug.DrawRay(transform.position, transform.right * visibleDistance, Color.red);

        // Raycasts
        RaycastHit hit;
        float fDist = 0, rDist = 0, lDist = 0, r45Dist = 0, l45Dist = 0;

        // forward
        if (Physics.Raycast(transform.position, transform.forward, out hit, visibleDistance))
        {
            fDist = 1 - Round(hit.distance / visibleDistance);
        }

        //right
        if (Physics.Raycast(transform.position, transform.right, out hit, visibleDistance))
        {
            rDist = 1 - Round(hit.distance / visibleDistance);
        }

        //left
        if (Physics.Raycast(transform.position, -transform.right, out hit, visibleDistance))
        {
            lDist = 1 - Round(hit.distance / visibleDistance);
        }

        //right 45
        if (Physics.Raycast(transform.position, Quaternion.AngleAxis(45, Vector3.up) * transform.right, out hit, visibleDistance))
        {
            r45Dist = 1 - Round(hit.distance / visibleDistance);
        }

        //left 45
        if (Physics.Raycast(transform.position, Quaternion.AngleAxis(45, Vector3.up) * -transform.right, out hit, visibleDistance))
        {
            l45Dist = 1 - Round(hit.distance / visibleDistance);
        }

        string td = fDist + "," + rDist + "," + lDist + "," +
            r45Dist + "," + l45Dist + "," + Round(translationInput) + "," + Round(rotationInput);

        // write training data to file
        // only write unique values to data
        if (!collectedTrainingData.Contains(td))
            collectedTrainingData.Add(td);
    }
}