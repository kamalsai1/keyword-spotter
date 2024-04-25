import React, { useState, useRef, useEffect } from "react";
import styles from "./AudioRecorder.module.css";
import WaveSurfer from "wavesurfer.js";

import axios from "axios";

function AudioRecorder() {
  const [audioURL, setAudioURL] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [timer, setTimer] = useState(0);
  const [intervalId, setIntervalId] = useState(null);
  // const audioInputRef = useRef();
  const mediaRecorderRef = useRef();
  const [submittedAudioURL, setSubmittedAudioURL] = useState("");
  const [audioBlob, setAudioBlob] = useState(null);

  const startRecording = async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        mediaRecorderRef.current = new MediaRecorder(stream);
        mediaRecorderRef.current.start();
        setIsRecording(true);
        startTimer();

        const audioChunks = [];
        mediaRecorderRef.current.ondataavailable = function (event) {
          audioChunks.push(event.data);
        };

        mediaRecorderRef.current.onstop = function () {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" }); // Ensure you set the correct MIME type
          const audioUrl = URL.createObjectURL(audioBlob);
          setAudioURL(audioUrl);
          setAudioBlob(audioBlob); // Store the Blob in state
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
    e.preventDefault(); // Prevent the form from actually submitting
    setSubmittedAudioURL(audioURL); // Set the submitted audio URL
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
      setAudioBlob(file); // Assuming 'file' is the Blob/File object
    }
  };

  const waveformRef = useRef(null);
  const [waveSurfer, setWaveSurfer] = useState(null);

  useEffect(() => {
    const ws = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: "#007bff",
      progressColor: "#0056b3",
      cursorColor: "transparent",
      barWidth: 2,
      barRadius: 3,
      responsive: true,
      height: 100,
      normalize: true,
      hideScrollbar: true,
    });

    ws.on("ready", () => {
      console.log("WaveSurfer is ready");
      ws.play();
    });
    ws.on("error", (e) => {
      console.error("WaveSurfer error:", e);
    });

    setWaveSurfer(ws);

    return () => ws && ws.destroy();
  }, []); // Removed dependency on waveSurfer

  useEffect(() => {
    if (waveSurfer && submittedAudioURL) {
      waveSurfer.load(submittedAudioURL);
      waveSurfer.on("ready", () => waveSurfer.play());
    }
  }, [submittedAudioURL, waveSurfer]);
  const handleWaveform = () => {
    console.log("handleWaveform called");

    if (waveSurfer && submittedAudioURL) {
      console.log("loading url:", submittedAudioURL);
      waveSurfer.load(submittedAudioURL);

      waveSurfer.once("ready", () => {
        console.log("waveform ready");
        waveSurfer.play();
      });

      window.dispatchEvent(new Event("resize"));
    }
  };

  const spotKeyword = () => {
    if (!submittedAudioURL) {
      console.log("No audio file");
      return;
    }

    // Assuming `audioBlob` is accessible and is the actual blob of the audio file
    const formData = new FormData();
    formData.append("audio", audioBlob, 'file'); // Make sure 'audioBlob' is the Blob of your audio file
    axios.post('http://localhost:5000/model', formData)
  .then(response => console.log(response.data))
  .catch(error => console.error('Error uploading the file', error));
  };

  return (
    <div className={styles.audioRecorderContainer}>
      <h2 style={{ textAlign: "center" }}>Record or Select an Audio File</h2>
      <div className={styles.buttonsContainer}>
        {isRecording ? (
          <button onClick={stopRecording} disabled={!isRecording}>
            Stop Recording
          </button>
        ) : (
          <button onClick={startRecording} disabled={isRecording}>
            Start Recording
          </button>
        )}
        <span className={styles.timer}>Timer: {timer}s</span>
      </div>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="audio/*"
          onChange={handleAudioFileChange}
          className={styles.audioInput}
        />
        <button type="submit">Submit</button>
      </form>
      {submittedAudioURL && (
        <div className={styles.audioPlayer}>
          <audio controls src={submittedAudioURL}></audio>
        </div>
      )}

      {/* New buttons for Waveform and Spot Keyword */}
      <div className={styles.buttonsContainer}>
        <button className={styles.waveformButton} onClick={handleWaveform}>
          Waveform
        </button>
        <button className={styles.spotKeywordButton} onClick={spotKeyword}>
          Spot Keyword
        </button>
      </div>

      {/* Waveform container */}
      <div id="waveform" ref={waveformRef}></div>
    </div>
  );
}

export default AudioRecorder;