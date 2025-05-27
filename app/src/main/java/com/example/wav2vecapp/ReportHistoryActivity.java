package com.example.wav2vecapp;

import android.app.DatePickerDialog;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

public class ReportHistoryActivity extends AppCompatActivity {

    Button btnStartDate, btnEndDate, btnSearch;
    Button btnBack;
    EditText etKeyword;
    RecyclerView recyclerView;
    ReportAdapter adapter;
    List<ReportItem> reportList = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_history);

        btnStartDate = findViewById(R.id.btnStartDate);
        btnEndDate = findViewById(R.id.btnEndDate);
        btnSearch = findViewById(R.id.btnSearch);
        etKeyword = findViewById(R.id.etKeyword);
        recyclerView = findViewById(R.id.recyclerViewReports);
        btnBack = findViewById(R.id.history_btnBack);

        adapter = new ReportAdapter(reportList);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);

        // Date pickers
        btnStartDate.setOnClickListener(v -> showDatePicker(btnStartDate));
        btnEndDate.setOnClickListener(v -> showDatePicker(btnEndDate));

        btnSearch.setOnClickListener(v -> {
            String keyword = etKeyword.getText().toString();
            String startDate = btnStartDate.getText().toString();
            String endDate = btnEndDate.getText().toString();
            fetchReportHistory(startDate, endDate, keyword);
        });

        btnBack.setOnClickListener(v-> finish());
    }

    private void showDatePicker(Button targetButton) {
        final Calendar calendar = Calendar.getInstance();
        int year = calendar.get(Calendar.YEAR);
        int month = calendar.get(Calendar.MONTH); // 0~11
        int day = calendar.get(Calendar.DAY_OF_MONTH);

        DatePickerDialog dialog = new DatePickerDialog(
                this,
                (view, y, m, d) -> {
                    // 날짜 선택 후 버튼 텍스트 업데이트
                    String selectedDate = String.format("%04d-%02d-%02d", y, m + 1, d);
                    targetButton.setText(selectedDate);
                },
                year, month, day
        );
        dialog.show();
    }

    private void fetchReportHistory(String start, String end, String keyword) {
        // TODO: 서버에서 start, end, keyword 기준으로 데이터 요청 및 reportList 갱신
        // 예시: Retrofit 호출 → 응답 결과 → reportList.clear(); reportList.addAll(...); adapter.notifyDataSetChanged();
    }
}

