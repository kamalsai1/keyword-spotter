import React, { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import axios from "axios";

function AudioRecorder() {
  const [audioURL, setAudioURL] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [timer, setTimer] = useState(0);
  const [intervalId, setIntervalId] = useState(null);
  const mediaRecorderRef = useRef();
  const [submittedAudioURL, setSubmittedAudioURL] = useState("");
  const [audioBlob, setAudioBlob] = useState(null);
  const waveformRef = useRef(null);
  const [waveSurfer, setWaveSurfer] = useState(null);
  const [prediction, setPrediction] = useState("");

  const startRecording = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        mediaRecorderRef.current.start();
        setIsRecording(true);
        startTimer();

        const audioChunks = [];
        mediaRecorderRef.current.ondataavailable = function (event) {
          audioChunks.push(event.data);
        };

        mediaRecorderRef.current.onstop = function () {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(audioBlob);
          setAudioURL(audioUrl);
          setAudioBlob(audioBlob);
        };
      } catch (err) {
        console.error("Error accessing the microphone:", err);
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      stopTimer();
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmittedAudioURL(audioURL);
    console.log(audioURL);
  };

  const startTimer = () => {
    setTimer(0);
    const id = setInterval(() => {
      setTimer((prevTime) => prevTime + 1);
    }, 1000);
    setIntervalId(id);
  };

  const stopTimer = () => {
    clearInterval(intervalId);
    setIntervalId(null);
  };

  const handleAudioFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioURL(URL.createObjectURL(file));
      setAudioBlob(file);
    }
  };

  useEffect(() => {
    const ws = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: "#007bff",
      progressColor: "#0056b3",
      cursorColor: "transparent",
      barWidth: 2,
      height: 100,
      normalize: true,
      responsive: true,
    });

    setWaveSurfer(ws);

    return () => ws && ws.destroy();
  }, [audioBlob]);

  useEffect(() => {
    if (waveSurfer && submittedAudioURL) {
      waveSurfer.load(submittedAudioURL);
      waveSurfer.on("ready", () => waveSurfer.play());
    }
  }, [submittedAudioURL, waveSurfer]);

  const spotKeyword = async () => {
    if (!submittedAudioURL) {
      console.log("No audio file");
      return;
    }

  const audioBlobToWav = async (blob) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          resolve(reader.result);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    };


    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "file");
      const response = await axios.post("https://keyword-spotter-backend-plim5ykzmpjkwerfaqqhhd.streamlit.app/model", formData);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error uploading the file", error);
    }
  };

  return (
    <div>
      <h2>Record or Select an Audio File</h2>
      <div>
        {isRecording ? (
          <button onClick={stopRecording} disabled={!isRecording}>
            Stop Recording
          </button>
        ) : (
          <button onClick={startRecording} disabled={isRecording}>
            Start Recording
          </button>
        )}
        <span>Timer: {timer}s</span>
      </div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="audio/*" onChange={handleAudioFileChange} />
        <button type="submit">Submit</button>
      </form>
      {submittedAudioURL && (
        <div>
          <audio controls src={submittedAudioURL}></audio>
        </div>
      )}

      <div>
        <button onClick={spotKeyword}>Spot Keyword</button>
        {prediction && <p>Prediction: {prediction}</p>}
      </div>

      <div ref={waveformRef}></div>+
      .
    </div>
  );
}

export default AudioRecorder;