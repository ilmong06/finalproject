package com.example.wav2vecapp;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import okhttp3.logging.HttpLoggingInterceptor;
import okhttp3.MediaType;

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
    private String uuid;
    private SharedPreferences sharedPreferences;
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

        sharedPreferences = getSharedPreferences("user_info", MODE_PRIVATE);
        uuid = sharedPreferences.getString("uuid", "");
        Log.d("UUID", "📌 UUID 불러오기 결과: " + uuid);

        /// get Keywords
        ApiService apiService = RetrofitClient.getClient().create(ApiService.class);
        KeywordRequest rq = new KeywordRequest(uuid);
        Call<List<KeywordItem>> call = apiService.getKeywordList(rq);

        call.enqueue(new Callback<List<KeywordItem>>() {
            @Override
            public void onResponse(Call<List<KeywordItem>> call, Response<List<KeywordItem>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    List<KeywordItem> keywordList = response.body();

                    // ✅ k_list를 비우고 키워드 재출력
                    LinearLayout kListLayout = findViewById(R.id.k_list);
                    kListLayout.removeAllViews();

                    for (KeywordItem item : keywordList) {
                        addKeywordToList(item.getKeyword());
                    }
                }
            }

            @Override
            public void onFailure(Call<List<KeywordItem>> call, Throwable t) {
                Toast.makeText(KeywordActivity.this, "키워드 불러오기 실패", Toast.LENGTH_SHORT).show();
            }
        });


        /// 뒤로가기 버튼
        btnBack.setOnClickListener(v -> finish());


        /// 키워드 등록
        btnAddKeyword.setOnClickListener(v -> {
            String keyword = etKeyword.getText().toString().trim();
            if (keyword.isEmpty()) {
                Toast.makeText(this, "❗ 키워드를 입력하세요.", Toast.LENGTH_SHORT).show();
                return;
            }
            keywordOrder++;
            addKeywordToList(keyword);
            sendKeywordToServer(keyword, keywordOrder);
        });

        /// 편집 버튼
        edit.setOnClickListener(v -> {
            isEditMode = !isEditMode;
            edit.setText(isEditMode ? "완료" : "편집");
            delete.setVisibility(isEditMode ? View.VISIBLE : View.GONE);
            toggleCheckBoxes(isEditMode);
        });

        delete.setOnClickListener(v -> deleteKeywords());
    }

    private void toggleCheckBoxes(boolean visible) {
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

    private void addKeywordToList(String keyword) {
        LinearLayout newItemLayout = new LinearLayout(this);
        newItemLayout.setOrientation(LinearLayout.HORIZONTAL);
        newItemLayout.setGravity(Gravity.CENTER_VERTICAL);
        newItemLayout.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));

        TextView keywordText = new TextView(this);
        keywordText.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 7));
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

    private void sendKeywordToServer(String keyword, int order) {
        RequestBody uuidBody = RequestBody.create(MediaType.parse("text/plain"), uuid);  // UUID가 먼저
        RequestBody keywordBody = RequestBody.create(MediaType.parse("text/plain"), keyword);
        RequestBody orderBody = RequestBody.create(MediaType.parse("text/plain"), String.valueOf(order));

// Retrofit 객체 생성
        Retrofit retrofit = getRetrofitClient();
        ApiService apiService = retrofit.create(ApiService.class);

// 전송
        Call<ResponseBody> call = apiService.registerKeyword(uuidBody, keywordBody, orderBody);

        call.enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<ResponseBody> call, Response<ResponseBody> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(KeywordActivity.this, "✅ 등록 완료", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(KeywordActivity.this, "❌ 서버 오류", Toast.LENGTH_SHORT).show();
                }
            }

            @Override
            public void onFailure(Call<ResponseBody> call, Throwable t) {
                Toast.makeText(KeywordActivity.this, "🚫 통신 실패: " + t.getMessage(), Toast.LENGTH_SHORT).show();
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

    private void deleteKeywords() {
        for (int i = layoutKeywordList.getChildCount() - 1; i >= 0; i--) {
            View view = layoutKeywordList.getChildAt(i);
            if (view instanceof LinearLayout row) {
                CheckBox checkBox = (CheckBox) row.getChildAt(2);
                if (checkBox.isChecked()) {
                    layoutKeywordList.removeViewAt(i);
                }
            }
        }
    }
}
