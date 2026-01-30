using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using System.Collections.Generic;

//send data to controller
public class socketCore : MonoBehaviour
{
    private Socket _clientSocket;
    private byte[] _recieveBuffer = new byte[8142];
    private IPEndPoint _serverEndPoint;
    public byte data;

    public List<byte> sendData = new List<byte>();
    byte[] sendDataUlt; // 声明一个byte数组变量
    // 开始连接
    void Start()
    {
        _clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        _serverEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 1233);  // 注意修改为你的服务器地址和端口
        _clientSocket.BeginConnect(_serverEndPoint, new AsyncCallback(ConnectCallback), null);
        //InvokeRepeating("SendMessage", 0, 0.001f); // 每秒发送一次数据
    }

    // 连接成功回调
    private void ConnectCallback(IAsyncResult AR)
    {
        try
        {
            _clientSocket.EndConnect(AR);
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    }

    void LateUpdate(){
        // sendData.Add(0x01);
        // sendData.Add(0x00);
        // sendData.Add(0xFD);
        // sendData.Add(0xFF);
        // sendData.Add(0xFE);
        // sendData.Add(0x02);
        // sendData.Add(0xAB);
        // sendData.Add(0xCD);
        // sendData.Add(0xFF);
        // sendData.Add(0xFE);


        sendDataUlt = sendData.ToArray();
        _clientSocket.BeginSend(sendDataUlt, 0, sendDataUlt.Length, SocketFlags.None, new AsyncCallback(SendCallback), null);
        sendData.Clear();

    }
    // 发送消息

    // 发送消息回调
    private void SendCallback(IAsyncResult AR)
    {
        try
        {
            _clientSocket.EndSend(AR);
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    }
}
