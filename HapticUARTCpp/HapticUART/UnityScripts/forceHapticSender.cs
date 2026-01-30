using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SprCs;

// send force to one finger
public class forceHapticSender : MonoBehaviour
{
    public byte motorNum = 0;
    socketCore core;
    [Range(0, 255)]
    short forceNum;

    public static PHSceneControl.rbContactValue myContact;
    PHSceneIf phScene;
    PHSolidBehaviour mySolid;
    void Start()
    {
        GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();

        foreach(GameObject obj in allObjects)
        {
            socketCore script = obj.GetComponent<socketCore>();
            if(script != null)
            {
                core = script;
            }

            PHSceneBehaviour phb = obj.GetComponent<PHSceneBehaviour>();
            if(phb != null)
            {
                phScene = phb.phScene;
            }
        }

        myContact = new PHSceneControl.rbContactValue();
        myContact.resetValues();
        mySolid = this.GetComponent<PHSolidBehaviour>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        
        // change forceNum here
        GetSumForce();
        var mappedValue = 255 * Mathf.InverseLerp(100, 500, Mathf.Abs(myContact.sumForce));
        forceNum = (byte)mappedValue;
        sendForce(); //其实不用一直发，但是目前没有性能问题 暂时
    }

    void sendForce(){
        
        List<byte> myData = new List<byte>();
        myData.Add(0x01);
        myData.Add(motorNum);
        myData.Add(0x00);
        myData.Add((byte)forceNum);
        myData.Add(0xFF);
        myData.Add(0xFE);
        lock (core.sendData)
        {
            core.sendData.AddRange(myData);
        }
        
    }

    void GetSumForce(){
        myContact.resetValues();
        Vec3d f = new Vec3d();
        Vec3d to = new Vec3d();
        Posed pose = new Posed();
        Vec3d fsum = new Vec3d();
        f = new Vec3d(0, 0, 0);
        to = new Vec3d(0, 0, 0);

        float relativeLinearSpeed = 0;
        for (int i = 0; i < phScene.NContacts(); i++){
                
                PHContactPointIf contact = phScene.GetContact(i);
                if(contact.GetSocketSolid() == mySolid.phSolid || contact.GetPlugSolid() == mySolid.phSolid){
                    contact.GetRelativeVelocity(f, to);
                    relativeLinearSpeed = (float)(f.x + f.y + f.z);
                    myContact.relativeAngularVelocity = (float)(to.x + to.y + to.z);
                    myContact.relativeLinearVelocity = relativeLinearSpeed;
                    myContact.isStaticFriction = contact.IsStaticFriction();
                    
                    contact.GetConstraintForce(f, to);
                    if(contact.GetSocketSolid() == mySolid.phSolid){
                        myContact.contactName = contact.GetPlugSolid().GetName();
                        
                        fsum -= pose.Ori() * f;
                        //tsum -= pose.Pos() % pose.Ori() * f;//扭矩
                    }
                    if(contact.GetPlugSolid() == mySolid.phSolid){
                        myContact.contactName = contact.GetSocketSolid().GetName();
                        contact.GetPlugPose(pose);
                        fsum += pose.Ori() * f;
                        //tsum += pose.Pos() % pose.Ori() * f;
                    }
                    myContact.sumForce = (Mathf.Abs((float)fsum.x) + Mathf.Abs((float)fsum.y) + Mathf.Abs((float)fsum.z));

                }
                
                
        }
    }
}

// public class rbContactValue{
//         public float relativeLinearVelocity = 0;
//         public float relativeAngularVelocity = 0;
//         public float sumForce = 0;
//         public string contactName = "";
//         public bool isStaticFriction = false;

//         public void resetValues(){
//             relativeLinearVelocity = 0;
//             relativeAngularVelocity = 0;
//             sumForce = 0;
//             contactName = "";
//             isStaticFriction = false;
//         }
// }