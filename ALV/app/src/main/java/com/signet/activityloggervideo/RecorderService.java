package com.signet.activityloggervideo;

import android.app.Service;
import android.content.Intent;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.hardware.Camera.CameraInfo;
import android.view.WindowManager;
import android.media.MediaRecorder;
import android.os.Environment;
import android.os.IBinder;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.Surface;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.media.MediaPlayer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import android.content.Context;

public class RecorderService extends Service {

    private static final String TAG = "RecorderService";
    private SurfaceView mSurfaceView;
    private SurfaceHolder mSurfaceHolder;
    private static Camera mServiceCamera;
    private boolean mRecordingStatus;
    private MediaRecorder mMediaRecorder;
    int rotation;

    long start_rec;
    long video_length;
    double fps = 30;
    double n_frames;
    long[] frame_timestamps;
    float focal_length;

    // variabili di MainActivity
    String video_path;
    //int cameraID;
    MediaPlayer mp = null;

    @Override
    public void onCreate() {
        mRecordingStatus = false;
        mServiceCamera = MainActivity.mCamera;
        mSurfaceView = MainActivity.mSurfaceView;
        mSurfaceHolder = MainActivity.mSurfaceHolder;
        mp = MediaPlayer.create(this, R.raw.beep);

        super.onCreate();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override // lanciato dopo onCreate()
    public int onStartCommand(Intent intent, int flags, int startId) {
        super.onStartCommand(intent, flags, startId);

        // recupero variabili da MainActivity
        video_path = intent.getStringExtra("video_path");
        //cameraID = intent.getIntExtra("cameraID", 0); // 0: back camera(default), 1: front camera

        if (mRecordingStatus == false) {
            startRecording();
            mp.start();
            mRecordingStatus = true;
        }

        return START_STICKY; // usato per servizi che hanno START/STOP esplicito
    }

    @Override
    public void onDestroy() {
        stopRecording();
        mp.start();
        mRecordingStatus = false;
        super.onDestroy();
    }

    public boolean startRecording(){
        try {

            mServiceCamera = Camera.open(getCameraBackId());

            Camera.Parameters p = mServiceCamera.getParameters(); // parametri camera
            p.setFocusMode(Camera.Parameters.FOCUS_MODE_INFINITY); // usato per acquisire frame per camera calibration

            // Check what PREVIEW sizes are supported by your camera
            final List<Size> listPreviewSize = p.getSupportedPreviewSizes();
            for (Size size : listPreviewSize) {
                Log.i(TAG, String.format("Supported Preview Size (%d, %d)", size.width, size.height));
            }

            Size previewSize = p.getPreferredPreviewSizeForVideo(); // Returns the preferred or recommended preview size
            if (previewSize == null){ // may return null
                previewSize = listPreviewSize.get(0);
            }

            Log.i(TAG, String.format("Chosen Preview Size (%d, %d)", previewSize.width, previewSize.height));
            p.setPreviewSize(previewSize.width, previewSize.height); // set the preview size

            mServiceCamera.setDisplayOrientation(getPreviewOrientation(RecorderService.this, getCameraBackId()));

            try {
                mServiceCamera.setPreviewDisplay(mSurfaceHolder);
                mServiceCamera.startPreview();
            }
            catch (IOException e) {
                Log.e(TAG, e.getMessage());
                e.printStackTrace();
            }

            focal_length = p.getFocalLength(); // camera focal length
            mServiceCamera.setParameters(p);

            mMediaRecorder = new MediaRecorder();
            mMediaRecorder.setCamera(mServiceCamera);
            //mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC); SCOMMENTARE PER OTTENERE AUDIO
            //mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.CAMERA);
            mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mMediaRecorder.setVideoFrameRate((int)fps); // stabilisce fps

            // Check what RESOLUTIONS are supported by your camera
            List<Size> listSize = getSupportedVideoSizes(mServiceCamera);
             for (Size size : listSize) {
                    Log.i(TAG, "Available resolution: " + size.width + " " + size.height);
                }
            int chosenQuality = 5; // 0: best possible resolution
            Size size = listSize.get(chosenQuality);
            Log.i(TAG, "Chosen resolution: " + size.width + " " + size.height);

            mMediaRecorder.setVideoSize(size.width, size.height); // setta la qualità di acquisizione video
            mMediaRecorder.setVideoEncodingBitRate(3000000);
            mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
            mMediaRecorder.setOutputFile(Environment.getExternalStorageDirectory().getPath() + "/ActivityLoggerVideo/" + video_path + "/" + video_path + ".mp4");
            mMediaRecorder.setPreviewDisplay(mSurfaceHolder.getSurface());

            rotation = getPreviewOrientation(RecorderService.this, getCameraBackId());

            mMediaRecorder.setOrientationHint(rotation);

            mServiceCamera.unlock(); // allow media process access the camera
            mMediaRecorder.prepare();

            mMediaRecorder.start();
            start_rec = System.nanoTime(); // inizio registrazione

            Log.i(TAG, "Recording STARTS at: " + String.valueOf(start_rec));

            mRecordingStatus = true;

            return true;

        } catch (IllegalStateException e) {
            Log.d(TAG, e.getMessage());
            e.printStackTrace();
            return false;

        } catch (IOException e) {
            Log.d(TAG, e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    public void stopRecording() {
        try {
            mServiceCamera.reconnect();
        }
        catch (IOException e) {
            e.printStackTrace();
        }

        mMediaRecorder.stop();
        long stop_rec = System.nanoTime();

        Log.i(TAG, "Recording STOPS at: " + String.valueOf(stop_rec));

        mMediaRecorder.reset();
        mServiceCamera.stopPreview();
        mMediaRecorder.release();
        mServiceCamera.release();
        mServiceCamera = null;

        // Retrieve video information
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(RecorderService.this, Uri.fromFile(new File(Environment.getExternalStorageDirectory().getPath() + "/ActivityLoggerVideo/" + video_path + "/" + video_path + ".mp4")));
        video_length = Long.parseLong(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION));

        n_frames = Math.round(video_length / ((1 / fps) * 1e3));

        // Compute frame timestaps asssuming uniform sampling frequency
        frame_timestamps = new long[(int)n_frames];
        for(int i = 0; i <= frame_timestamps.length-1; i++)
             frame_timestamps[i] = (long) ((start_rec) + i * (1/fps) * 1e9);

        writeFrameTimestamps(); // write timestamp in a file
        writeVideoInfo(); // duration and number of video frames are written in a file

    }

    public static int getDeviceOrientation(Context context) {

        int degrees = 0;
        WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        int rotation = windowManager.getDefaultDisplay().getRotation();

        switch(rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        return degrees;
    }

    public static int getPreviewOrientation(Context context, int cameraId) {

        int temp = 0;
        int previewOrientation = 0;

        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
        Camera.getCameraInfo(cameraId, cameraInfo);

        int deviceOrientation = getDeviceOrientation(context);
        temp = cameraInfo.orientation - deviceOrientation + 360;
        previewOrientation = temp % 360;

        return previewOrientation;
    }

    // Returns rear-facing camera ID
    public int getCameraBackId(){

        int numberOfCameras = Camera.getNumberOfCameras();

        CameraInfo cameraInfo = new CameraInfo();
        for (int i = 0; i < numberOfCameras; i++) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == CameraInfo.CAMERA_FACING_BACK) {
                return i;
            }
        }
        return -1;// Device do not have back camera
    }

    /*
    // Restituisce ID camera FRONTALE
    public int getCameraFrontId(){

        int numberOfCameras = Camera.getNumberOfCameras();

        CameraInfo cameraInfo = new CameraInfo();
        for (int i = 0; i < numberOfCameras; i++) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == CameraInfo.CAMERA_FACING_FRONT) {
                return i;
            }
        }
        return -1;// Device do not have front camera
    }
    */

    // Restituisce tutte le possibili risoluzioni per i video
    public List<Size> getSupportedVideoSizes(Camera camera) {
        if (camera.getParameters().getSupportedVideoSizes() == null || camera.getParameters().getSupportedVideoSizes().size() == 0)
        {   // Video sizes may be null, which indicates that all the supported
            // preview sizes are supported for video recording.
            return camera.getParameters().getSupportedPreviewSizes();
        }
        else
        {
            return camera.getParameters().getSupportedVideoSizes();
        }
    }

    public void writeVideoInfo(){
        String path = Environment.getExternalStorageDirectory().getPath() + "/ActivityLoggerVideo/" + video_path;
        File start_time = new File(path, "video_info.txt");

        try {
            FileOutputStream fos = new FileOutputStream(start_time);
            fos.write(("Start" + "\t" + start_rec + "\n" + "Length" + "\t" + video_length + "\n" +
                    "n_frames" + "\t" + n_frames + "\n" + "focal_length" + "\t" + focal_length + "\n").getBytes());
            fos.close();
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
    }

    public void writeFrameTimestamps(){
        String path = Environment.getExternalStorageDirectory().getPath() + "/ActivityLoggerVideo/" + video_path;
        File frame_timestamp_file = new File(path, "frame_timestamp.csv");

        try{
            FileOutputStream fos = new FileOutputStream(frame_timestamp_file,true);
            fos.write("frame_timestamp\n".getBytes());

            for(int i = 0; i <= frame_timestamps.length -1; i++)
            {
                fos.write((String.valueOf(frame_timestamps[i]) + "\n").getBytes());
            }
            fos.close();
        }
        catch(IOException e)
        {
            e.printStackTrace();
        }
    }

}
