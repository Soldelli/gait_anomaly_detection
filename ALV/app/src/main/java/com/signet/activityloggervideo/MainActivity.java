package com.signet.activityloggervideo;

import android.accounts.Account;
import android.accounts.AccountManager;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.AsyncTask;
import android.os.Environment;
import android.os.Handler;
import android.os.PowerManager;
import android.os.PowerManager.WakeLock;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.Switch;
import android.widget.CompoundButton;
import org.apache.commons.net.ftp.FTP;
import org.apache.commons.net.ftp.FTPClient;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class MainActivity extends ActionBarActivity implements SensorEventListener, SurfaceHolder.Callback {

    private static final String TAG = "MainActivity";

    Spinner spinner;
    TextView title;
    File file,main_dir,dir;
    String dir_name;
    EditText name_input, note, start_delay, acquisition_time;
    Calendar cal;
    Button start_button, stop_button;
    Switch enable_camera_switch;
    int duration;
    int enable_camera = 1; // 0: camera not enabled  1: camera enabled

    // fos for writing files
    FileOutputStream fos_acc;
    FileOutputStream fos_acclin;
    FileOutputStream fos_gyro;
    FileOutputStream fos_magn;
    FileOutputStream fos_grav;
    FileOutputStream fos_rotvec;

    // CAMERA
    public static Camera mCamera;
    public static SurfaceView mSurfaceView;
    public static SurfaceHolder mSurfaceHolder;

    // Cartella destinazione file di acquisizione
    String main_dir_name = "ActivityLoggerVideo";

    // SENSORI
    private SensorManager mSensorManager;
    private Sensor accel;
    private Sensor accel_lin;
    private Sensor gyro;
    private Sensor magn;
    private Sensor grav;
    private Sensor rotvec;

    // FILES DATI
    File accelFile;
    File accel_linFile;
    File gyroFile;
    File magnFile;
    File gravFile;
    File rotvecFile;

    String accelFileName;
    String accel_linFileName;
    String gyroFileName;
    String magnFileName;
    String gravFileName;
    String rotvecFileName;

    Intent intent_service;

    // Flag di stop
    private boolean flag;
    private boolean done;

    // Variabili per calcolare offset tra event.timestamp e System.nanoTime()
    long[] Systs = new long[1000];
    long[] Evts = new long[1000];
    int counter;
    long startoffset;
    double time_difference;

    // Power manager for wakelock
    PowerManager powerManager;
    private WakeLock wakeLock;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        // Prende riferimenti dalle risorse
        title = (TextView)findViewById(R.id.textView);
        name_input = (EditText) findViewById(R.id.name_input);
        note = (EditText) findViewById(R.id.editText_note);
        start_delay = (EditText) findViewById(R.id.start_delay);
        acquisition_time = (EditText) findViewById(R.id.acquisition_time);
        start_button = (Button) findViewById(R.id.start_button);
        stop_button = (Button) findViewById(R.id.stop_button);
        stop_button.requestFocus();
        // camera stuff
        enable_camera_switch = (Switch) findViewById(R.id.enable_camera_switch); // Front camera switch button
        mSurfaceView = (SurfaceView) findViewById(R.id.surfaceView1);
        mSurfaceHolder = mSurfaceView.getHolder();
        mSurfaceHolder.addCallback(this);
        mSurfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        // Crea oggetti sensori
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accel = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        accel_lin = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        gyro = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        magn = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        grav = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        rotvec = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

        // Creo un wakelock (mantiene la CPU operativa durante l'acquisizione)
        powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,"Acquisition_Wakelock");

        // Popola il menu dropdown
        spinner = (Spinner)findViewById(R.id.spinner); // mi dice che attivita' ho selezionato
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,R.array.activities,android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        // Imposta l'ultimo nome utilizzato
        name_input.setText(retrieve_name());

        // Crea cartella di destinazione dei files, se non esiste
        create_folder();

        // enable camera switch button
        enable_camera_switch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean bChecked) {
                if (bChecked) {
                    enable_camera = 1; // front camera
                }
                else {
                    enable_camera = 0; // back camera
                }
            }
        });

        // Definisce callback pulsante START
        start_button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Creo intent per la Camera - se flaggata
                if (enable_camera == 1) {
                    intent_service = new Intent(MainActivity.this, RecorderService.class);
                    intent_service.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                }
                // Button enable
                start_button.setEnabled(false);
                stop_button.setEnabled(true);
                flag = true;
                done = false;
                counter = 0;
                startoffset = 0;
                time_difference = 0;

                // Salvo il nome in memoria non volatile
                save_name();
                // Creo cartella utente e i vari file per i sensori
                dir_name = spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time();
                dir = new File(main_dir, dir_name);
                dir.mkdirs();

                if(enable_camera == 1) {
                    // uso un intent per passare variabili a RecorderService
                    intent_service.putExtra("video_path", dir_name); // dove andra' salvato il video
                    //intent_service.putExtra("cameraID",cameraID); // front o back camera
                }
                // Creo i vari file .txt dove andranno salvati i dati acquisiti
                create_file_txt();
                // Creo i vari FileOutputStreams per i vari files
                create_fos();
                // Scrivo file NOTE specificato dall'utente
                write_note_file();

                // Verifica ritardo e durata acquisizione
                int delay = Integer.valueOf(start_delay.getText().toString());
                duration = Integer.valueOf(acquisition_time.getText().toString());

                final Handler handler_start = new Handler();
                final Handler handler_stop = new Handler();

                // Avvio il wakelock
                wakeLock.acquire();

                // Imposto eventualmente timer per l'interruzione dell'acquisizione
                if (duration>0) {
                    handler_stop.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            stop_button.callOnClick();
                        }
                    }, (duration+delay)*1000);
                }
                // Verifico acquisizione ritardata
                if (delay>0) {
                    handler_start.postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            start_acquisition(); // sensori + camera
                        }
                    }, delay*1000);
                }
                else {
                    // Fa partire l'acquisizione sensori e video
                   start_acquisition(); // sensori + eventuale camera
                }

            }
        });

        // Definisce callback pulsante STOP
        stop_button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Button enable
                start_button.setEnabled(true);
                stop_button.setEnabled(false);
                flag = false;
                done = false;
                // Stop acquisizione sensori e camera
                stop_acquisition(); // sensori + eventuale camera

                double mean_difference = time_difference / counter;
                //show("Average time difference: " + String.valueOf(mean_difference));
                if (mean_difference < 0.2){
                    //save_data();
                    show("Data saved!");}
                else
                    show("Bad acquisition, please retry.");
                // Rilascio il wakelock
                wakeLock.release();
            }
        });

    }

    //Abilita il menù del tasto opzioni
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.del_all_data) { // pulsante per cancellazione dati
            //delete_all_data(main_dir);
            show("All acquisition files have been deleted");
        }
        else if (id == R.id.send_data) { // pulsante per invio dati a server
            send_data();
        }

        return super.onOptionsItemSelected(item);
    }

    // Questa funzione viene chiamata ogni volta che uno dei sensori acquisisce un nuovo dato
    // il nuovo dato avra' un nuovo event.timestamp
    @Override
    public final void onSensorChanged(SensorEvent event)
    {
        if (flag) {
            // Calcolo offset tra event.timestamp e System.nanoTime()
            if(!done){
                Systs[counter] = System.nanoTime();
                Evts[counter] = event.timestamp;
                counter ++;
                if(counter == Systs.length) {
                    int samples = 10;
                    for(int i = Systs.length - samples; i <= Systs.length - 1; i++)
                        startoffset = startoffset + (Systs[i] - Evts[i]);
                    startoffset = startoffset / samples;
                    done = true;
                    counter = 0;
                }
            }
            else {
                //Prova test offset
                long time_ns = System.nanoTime();
                long sensorTimens = event.timestamp + startoffset;
                time_difference = time_difference + (Math.abs((sensorTimens - time_ns) * 1e-9));
                counter ++;

                Log.i(TAG,"Differenza di: " + String.valueOf((sensorTimens - time_ns) * 1e-9) + " s");

                // Salvo i dati del sensore in una variabile di tipo Float per aggiungerla alla corrispettiva lista
                Float[] temp = {event.values[0], event.values[1], event.values[2]};

                // Verifico che sensore ha generato i dati e salvo nel corrispettivo file
                switch (event.sensor.getType()) {
                    // scrivo valore del sensore nel file
                    case Sensor.TYPE_ACCELEROMETER:
                        write_file(temp, event.timestamp + startoffset, fos_acc);
                        break;
                    case Sensor.TYPE_LINEAR_ACCELERATION:
                        write_file(temp, event.timestamp + startoffset, fos_acclin);
                        break;
                    case Sensor.TYPE_GYROSCOPE:
                        write_file(temp, event.timestamp + startoffset, fos_gyro);
                        break;
                    case Sensor.TYPE_MAGNETIC_FIELD:
                        write_file(temp, event.timestamp + startoffset, fos_magn);
                        break;
                    case Sensor.TYPE_GRAVITY:
                        write_file(temp, event.timestamp + startoffset, fos_grav);
                        break;
                    case Sensor.TYPE_ROTATION_VECTOR:
                        write_file(temp, event.timestamp + startoffset, fos_rotvec);
                        break;
                }
            }
        }

    }

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy)
    {
        // Do something here if sensor accuracy changes.
    }

    // Avvia i servizi per l'acquisizione dei dati dai sensori
    private void start_acquisition()
    {
        // Avvio i servizi per l'acquisizione dati dai vari sensori
        mSensorManager.registerListener(MainActivity.this, accel, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(MainActivity.this, accel_lin, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(MainActivity.this, gyro, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(MainActivity.this, magn, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(MainActivity.this, grav, SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(MainActivity.this, rotvec, SensorManager.SENSOR_DELAY_FASTEST);
        if (enable_camera == 1) {
            startService(intent_service); // start camera service
        }
        // Visualizza avviso a schermo
        show("START acquisition");
    }

    // Interrompe i servizi per l'acquisizione dei dati dai sensori
    private void stop_acquisition() {
        // Interrompo i servizi precedentemente avviati
        mSensorManager.unregisterListener(MainActivity.this); // sensori
        if (enable_camera == 1) {
            stopService(new Intent(MainActivity.this, RecorderService.class)); // camera
        }
        // close all FileOutputStreams
        close_fos();
    }

    // Crea file .txt per i dati acquisiti
    private void create_file_txt(){
        accelFileName = "accellerometer_" + spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time() + ".csv";
        accelFile = new File(dir, accelFileName);
        write_incipit(accelFile, "acc"); // scrive prima riga

        accel_linFileName = "linearaccellerometer_" + spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time() + ".csv";
        accel_linFile = new File(dir, accel_linFileName);
        write_incipit(accel_linFile, "acc_lin");

        gyroFileName = "gyroscope_" + spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time() + ".csv";
        gyroFile = new File(dir, gyroFileName);
        write_incipit(gyroFile, "gyro");

        magnFileName = "magnetometer_" + spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time() + ".csv";
        magnFile = new File(dir, magnFileName);
        write_incipit(magnFile, "magn");

        gravFileName = "gravity_" + spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time() + ".csv";
        gravFile = new File(dir, gravFileName);
        write_incipit(gravFile, "grav");

        rotvecFileName = "rotvec_" + spinner.getSelectedItem().toString() + "_" + name_input.getText() + "_" + get_time() + ".csv";
        rotvecFile = new File(dir, rotvecFileName);
        write_incipit(rotvecFile, "rotvec");
    }

    // Crea FileOutputStream per ogni file creato
    private void create_fos(){
        try {
            fos_acc = new FileOutputStream(accelFile, true);
            fos_acclin = new FileOutputStream(accel_linFile, true);
            fos_gyro = new FileOutputStream(gyroFile, true);
            fos_magn = new FileOutputStream(magnFile, true);
            fos_grav = new FileOutputStream(gravFile, true);
            fos_rotvec = new FileOutputStream(rotvecFile, true);
        }
        catch(IOException e) {
            e.printStackTrace();}
    }

    // Chiude tutti i FileOutputStream utilizzati
    private void close_fos(){
        try {
            fos_acc.close();
            fos_acclin.close();
            fos_gyro.close();
            fos_magn.close();
            fos_grav.close();
            fos_rotvec.close();
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }

    // Salvare dati acquisiti?
    private void save_data()
    {
        // Inizializzo dialog
        AlertDialog.Builder dlgAlert = new AlertDialog.Builder(this);
        // Configuro dialog
        dlgAlert.setTitle("Acquisition completed");
        if(duration>0) {
            dlgAlert.setMessage("You acquired around " + Integer.toString(duration) + "s of " + spinner.getSelectedItem().toString() + " activity.\nDo you want to save data?");
        }
        else {
            dlgAlert.setMessage("You acquired data from " + spinner.getSelectedItem().toString() + " activity.\nDo you want to save data?");
        }
         dlgAlert.setPositiveButton("Yes",new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int whichButton) {
                show("Data saved!");
                // Button enable
                start_button.setEnabled(true);
                stop_button.setEnabled(false);
                flag = false;
                done = false;
            }
        });
        dlgAlert.setNegativeButton("No", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id) {
                // Do nothing
                delete_all_data(dir);
                show("Data NOT saved!");
            }
        });
        dlgAlert.show();
    }

    // Crea la cartella dove andranno salvati i file
    private void create_folder()
    {
        main_dir = new File(Environment.getExternalStorageDirectory(), main_dir_name);
        if (!main_dir.exists()) {
            if (!main_dir.mkdirs()) {
                show("SD card error");
            }
        }
    }

    // Salva il nome contenuto della EditBox in memoria non volatile
    private void save_name()
    {
        SharedPreferences pref = getPreferences(Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = pref.edit();
        editor.putString("last_name", name_input.getText().toString());
        editor.apply();
    }

    // Restituisce il nome utilizzato nell'ultima acquisizione
    private String retrieve_name()
    {
        SharedPreferences pref = getPreferences(Context.MODE_PRIVATE);
        return pref.getString("last_name","Name");
    }

    // Stampa un avviso a schermo
    private void show(String string)
    {
        Toast.makeText(getApplicationContext(),string, Toast.LENGTH_SHORT).show();
    }

    // Scrive la prima riga del file
    private void write_incipit(File file, String data_type)
    {
        FileOutputStream fos;
        try {
            fos = new FileOutputStream(file, true);
            fos.write((data_type + "_timestamp\t" + data_type + "_x\t" + data_type + "_y\t" + data_type + "_z\n").getBytes());
            fos.close();
        }
        catch(IOException e)
        {
            e.printStackTrace();
            // Visualizza avviso a schermo
            show("ERROR while saving file!!");
        }
    }

    // Genera un file contenente i dati passati come parametri
    private void write_file(Float[] temp, Long timestamp, FileOutputStream fos)
    {
        try
        {
            fos.write((String.valueOf(timestamp) + "\t" + String.valueOf(temp[0].floatValue()) + "\t" + String.valueOf(temp[1].floatValue()) + "\t" + String.valueOf(temp[2].floatValue()) + "\n").getBytes());
        }
        catch (Exception e)
        {
            e.printStackTrace();
            // Visualizza avviso a schermo
            show("ERROR while saving file!!");
        }
    }

    // Restituisce una stringa contenente data e ora attuale
    private String get_time()
    {
        // Recupero la data attuale
        cal = Calendar.getInstance();
        String time,year,month,day,hour,min,sec;
        year = String.valueOf(cal.get(Calendar.YEAR));
        month = String.valueOf(cal.get(Calendar.MONTH) + 1);
        day = String.valueOf(cal.get(Calendar.DAY_OF_MONTH));
        hour = String.valueOf(cal.get(Calendar.HOUR_OF_DAY));
        min = String.valueOf(cal.get(Calendar.MINUTE));
        sec = String.valueOf(cal.get(Calendar.SECOND));

        // Rendo tutte le variabili lunghe 2 caratteri
        if (month.length()==1)
            month = "0" + month;
        if (day.length()==1)
            day = "0" + day;
        if (hour.length()==1)
            hour = "0" + hour;
        if (min.length()==1)
            min = "0" + min;
        if (sec.length()==1)
            sec = "0" + sec;

        time = "[" + day + "-" + month + "-" + year + "]_[" + hour + "." + min + "." + sec + "]";

        return time;

    }

    // Scrive file di note spicificate dall'utente
    private void write_note_file()
    {
        // Creo il file
        file = new File(dir,"note.txt");

        // Riempio il file
        FileOutputStream fos;
        try {
            fos = new FileOutputStream(file);
            fos.write((note.getText().toString()).getBytes());
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
            // Visualizza avviso a schermo
            show("ERROR while saving file!!");
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {

    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    private List<File> getListFiles(File parentDir) {
        ArrayList<File> inFiles = new ArrayList<File>();
        File[] files = parentDir.listFiles();
        for (File file : files) {
            if (file.isDirectory()) {
                inFiles.addAll(getListFiles(file));
            } else {
                if (!file.getName().endsWith(".zip"))
                    inFiles.add(file);
            }
        }
        return inFiles;
    }

    private String[] files_list_to_string(List<File> files)
    {
        String[] files_string = new String[files.size()];
        int i = 0;
        for (File file : files)
        {
            files_string[i] = file.getAbsolutePath();
            i += 1;
        }
        return files_string;
    }

    private static final int BUFFER = 2048;
    private void zip(String[] _files, String zipFileName) {
        try {
            BufferedInputStream origin = null;
            FileOutputStream dest = new FileOutputStream(zipFileName);
            ZipOutputStream out = new ZipOutputStream(new BufferedOutputStream(dest));
            byte data[] = new byte[BUFFER];
            String temp;

            for (int i = 0; i < _files.length; i++) {
                Log.v("Compress", "Adding: " + _files[i]);

                FileInputStream fi = new FileInputStream(_files[i]);
                origin = new BufferedInputStream(fi, BUFFER);

                temp = _files[i].substring(0,_files[i].lastIndexOf("/"));
                ZipEntry entry = new ZipEntry(_files[i].substring(temp.lastIndexOf("/") + 1));
                out.putNextEntry(entry);
                int count;

                while ((count = origin.read(data, 0, BUFFER)) != -1) {
                    out.write(data, 0, count);
                }
                origin.close();
            }

            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void send_data()
    {
        new compress_data().execute();
    }

    private void delete_zip_files()
    {
        File[] files = main_dir.listFiles();
        for (File file : files) {
            if (!file.isDirectory())
            {
                if (file.getAbsolutePath().endsWith(".zip"))
                    file.delete();
            }
        }
    }

    private String getUsername() {
        AccountManager manager = AccountManager.get(this);
        Account[] accounts = manager.getAccountsByType("com.google");
        List<String> possibleEmails = new LinkedList<String>();

        for (Account account : accounts) {
            // TODO: Check possibleEmail against an email regex or treat
            // account.name as an email address only for certain account.type values.
            possibleEmails.add(account.name);
        }

        if (!possibleEmails.isEmpty() && possibleEmails.get(0) != null) {
            String email = possibleEmails.get(0);
            String[] parts = email.split("@");

            if (parts.length > 1)
                return parts[0];
        }
        return "unknown_user";
    }

    private void ask_for_upload(final String data_to_send)
    {
        // Inizializzo dialog
        AlertDialog.Builder dlgAlert  = new AlertDialog.Builder(MainActivity.this);
        // Configuro dialog
        dlgAlert.setTitle("Send data");
        // Verifico dimensione dei dati
        File file = new File(data_to_send);
        long length = file.length();
        length = length/1024;
        dlgAlert.setMessage("Do you want to send your data to our server? It may require several minutes (Data to send: " + String.valueOf(length) + " KB)");
        dlgAlert.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int whichButton) {
                new send_file_FTP().execute(data_to_send);
            }
        });
        dlgAlert.setNegativeButton("No", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id) {
                show("Data NOT sended.");
            }
        });
        dlgAlert.create().show();
    }


    private void show_only_ok (String message)
    {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setMessage(message)
                .setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        //do things
                    }
                });
        AlertDialog alert = builder.create();
        alert.show();
    }

    private void delete_all_data(File folder) // cancella la cartella dell'ultima acquisizione
    {
        DeleteRecursive(folder); //"main_dir" per cancellare TUTTI i dati di tutte le acquisizioni
        create_folder();
    }

    void DeleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory())
            for (File child : fileOrDirectory.listFiles())
                DeleteRecursive(child);

        fileOrDirectory.delete();
    }

    private class send_file_FTP extends AsyncTask<String, Integer, Boolean> {

        //String server = "ftp.activitylogger.host-ed.me";
        //String user = "data@activitylogger.host-ed.me";
        //String pass = "Leone1992";
        //String serverRoad = "";

        String server = "37.182.61.38";
        String user = "pi";
        String pass = "Leone1992";
        String serverRoad = "fileTesi";

        /** The system calls this to perform work in a worker thread and
         * delivers it the parameters given to AsyncTask.execute() */
        protected Boolean doInBackground(String... file_path)
        {
            String filename = file_path[0].substring(file_path[0].lastIndexOf("/") + 1);

            try
            {
                FTPClient ftpClient = new FTPClient();
                //ftpClient.connect(InetAddress.getByName(server));
                //ftpClient.addProtocolCommandListener(new PrintCommandListener(new PrintWriter(System.out)));
                ftpClient.connect(server);
                ftpClient.login(user, pass);
                ftpClient.enterLocalPassiveMode();
                ftpClient.setFileType(FTP.BINARY_FILE_TYPE);

                ftpClient.changeWorkingDirectory(serverRoad);

                FileInputStream file = new FileInputStream(new File(file_path[0]));
                if (!ftpClient.storeFile(filename, file))
                {
                    file.close();
                    ftpClient.logout();
                    ftpClient.disconnect();
                    return false;
                }
                file.close();
                ftpClient.logout();
                ftpClient.disconnect();
                return true;
            }
            catch (Exception e)
            {
                e.printStackTrace();
                return false;
            }

        }

        ProgressDialog progress_dialog;
        protected  void onPreExecute()
        {
            progress_dialog = ProgressDialog.show(MainActivity.this, "Uploading data", "Please wait...\nBe patient. It may require several minutes.", true);
        }

        /** The system calls this to perform work in the UI thread and delivers
         * the result from doInBackground() */
        protected void onPostExecute(Boolean result)
        {
            progress_dialog.dismiss();

            if (result)
            {
                show_only_ok("Data sended.");
                delete_all_data(dir);
            }
            else
            {
                show_only_ok("ERROR! Data NOT sended.");
            }
        }

    }

    private class compress_data extends AsyncTask<String, Integer, String>
    {
        protected String doInBackground(String... file_path)
        {
            getUsername();
            delete_zip_files();
            List<File> files = getListFiles(main_dir);
            String[] file_list = files_list_to_string(files);

            String data_to_send = main_dir.getAbsolutePath()+"/"+getUsername()+"_all_data_"+get_time()+".zip";
            zip(file_list, data_to_send);

            return data_to_send;
        }

        ProgressDialog progress_dialog;
        protected  void onPreExecute()
        {
            progress_dialog = ProgressDialog.show(MainActivity.this, "Compressing data", "Please wait...", true);
        }

        protected void onPostExecute(String data_to_send)
        {
            progress_dialog.dismiss();
            ask_for_upload(data_to_send);
        }

    }

}
