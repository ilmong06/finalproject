package com.example.wav2vecapp;

import android.Manifest;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONObject;

import java.io.File;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class KeywordActivity extends AppCompatActivity {

    private EditText etKeyword;
    private Button btnAddKeyword, btnBack, delete, edit;
    private LinearLayout layoutKeywordList;
    private boolean isEditMode = false;

    private MediaRecorder recorder;
    private String filePath;
    private boolean isKeywordRegistering = false;
    private int registerCount = 0;
    private String currentKeyword = "";

    private SharedPreferences sharedPreferences;
    private String uuid;
    private int keywordOrder = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_keyword);

        etKeyword = findViewById(R.id.etKeyword);
        btnAddKeyword = findViewById(R.id.btnAddKeyword);
        btnBack = findViewById(R.id.btnBack);
        layoutKeywordList = findViewById(R.id.layout_keyword_list);
        delete = findViewById(R.id.btnDeleteSelected);
        edit = findViewById(R.id.editKeyword);

        filePath = getExternalFilesDir(null).getAbsolutePath() + "/keyword_recorded.wav";
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        uuid = sharedPreferences.getString("uuid", "");

        btnBack.setOnClickListener(v -> finish());

        btnAddKeyword.setOnClickListener(v -> {
            currentKeyword = etKeyword.getText().toString().trim();
            if (currentKeyword.isEmpty()) {
                Toast.makeText(this, "❗ 키워드를 입력하세요.", Toast.LENGTH_SHORT).show();
                return;
            }
            addKeywordToList(currentKeyword);
            startKeywordRegistration();
        });

        edit.setOnClickListener(v -> {
            isEditMode = !isEditMode;
            edit.setText(isEditMode ? "완료" : "편집");
            delete.setVisibility(isEditMode ? View.VISIBLE : View.GONE);
            checkBoxVis(isEditMode);
        });

        delete.setOnClickListener(v -> deleteKeywords());

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 100);
        }
    }

    private void checkBoxVis(boolean visible) {
        for (int i = 0; i < layoutKeywordList.getChildCount(); i++) {
            View view = layoutKeywordList.getChildAt(i);
            if (view instanceof LinearLayout l) {
                for (int j = 0; j < l.getChildCount(); j++) {
                    View child = l.getChildAt(j);
                    if (child instanceof CheckBox) {
                        child.setVisibility(visible ? View.VISIBLE : View.GONE);
                    }
                }
            }
        }
    }

    private void startRecording() {
        try {
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
            recorder.setOutputFile(filePath);

            recorder.prepare();
            recorder.start();
            Toast.makeText(this, "🎙️ 녹음 중...", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "❌ 녹음 실패", Toast.LENGTH_SHORT).show();
        }
    }

    private void stopRecording() {
        try {
            recorder.stop();
            recorder.release();
            recorder = null;

            Toast.makeText(this, "🎧 녹음 완료", Toast.LENGTH_SHORT).show();

            if (isKeywordRegistering) {
                sendAudioToKeywordRegister(filePath, currentKeyword, keywordOrder);
            }

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "❌ 녹음 중지 실패", Toast.LENGTH_SHORT).show();
        }
    }

    private void startKeywordRegistration() {
        isKeywordRegistering = true;
        registerCount = 0;
        keywordOrder++;
        Toast.makeText(this, "🔑 키워드 등록 시작 (1/6)", Toast.LENGTH_SHORT).show();
        startRecording();
    }

    private void sendAudioToKeywordRegister(String filePath, String keyword, int order) {
        File file = new File(filePath);
        RequestBody reqFile = RequestBody.create(MediaType.parse("audio/wav"), file);
        MultipartBody.Part body = MultipartBody.Part.createFormData("file", file.getName(), reqFile);
        RequestBody keywordBody = RequestBody.create(MediaType.parse("text/plain"), keyword);
        RequestBody uuidBody = RequestBody.create(MediaType.parse("text/plain"), uuid);
        RequestBody orderBody = RequestBody.create(MediaType.parse("text/plain"), String.valueOf(order));

        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);

        Call<ResponseBody> call = apiService.registerKeyword(keywordBody, uuidBody, orderBody);
        call.enqueue(new Callback<>() {
            @Override
            public void onResponse(@NonNull Call<ResponseBody> call, @NonNull Response<ResponseBody> response) {
                if (response.isSuccessful()) {
                    try {
                        JSONObject json = new JSONObject(response.body().string());
                        Toast.makeText(KeywordActivity.this, "✅ " + json.getString("message"), Toast.LENGTH_SHORT).show();
                        registerCount++;
                        if (registerCount < 6) {
                            startRecording();
                        } else {
                            isKeywordRegistering = false;
                            registerCount = 0;
                            Toast.makeText(KeywordActivity.this, "🔚 6회 등록 완료", Toast.LENGTH_LONG).show();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } else {
                    Toast.makeText(KeywordActivity.this, "❌ 등록 실패", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(@NonNull Call<ResponseBody> call, @NonNull Throwable t) {
                Toast.makeText(KeywordActivity.this, "🚫 오류: " + t.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    private Retrofit getRetrofitClient() {
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor();
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        OkHttpClient client = new OkHttpClient.Builder().addInterceptor(logging).build();

        return new Retrofit.Builder()
                .baseUrl(BuildConfig.BACKEND_BASE_URL)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
    }

    private void addKeywordToList(String keyword) {
        LinearLayout newItemLayout = new LinearLayout(this);
        newItemLayout.setOrientation(LinearLayout.HORIZONTAL);
        newItemLayout.setGravity(Gravity.CENTER_VERTICAL);
        newItemLayout.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

        TextView keywordText = new TextView(this);
        keywordText.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                7
        ));
        keywordText.setText(keyword);
        keywordText.setTextSize(20);

        View spacer = new View(this);
        spacer.setLayoutParams(new LinearLayout.LayoutParams(0, 1, 1));

        CheckBox checkBox = new CheckBox(this);
        checkBox.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 2));
        checkBox.setVisibility(View.GONE);

        newItemLayout.addView(keywordText);
        newItemLayout.addView(spacer);
        newItemLayout.addView(checkBox);

        layoutKeywordList.addView(newItemLayout);
    }

    private void deleteKeywords() {
        for (int i = layoutKeywordList.getChildCount() - 1; i >= 0; i--) {
            View view = layoutKeywordList.getChildAt(i);
            if (view instanceof LinearLayout row) {
                CheckBox checkBox = (CheckBox) row.getChildAt(2);
                if (checkBox.isChecked()) {
                    layoutKeywordList.removeViewAt(i);
                    // 서버 삭제 필요 시 추가 구현
                }
            }
        }
    }
}
