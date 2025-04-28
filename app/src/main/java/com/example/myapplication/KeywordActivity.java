package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class KeywordActivity extends AppCompatActivity {

    private EditText etKeyword;
    private Button btnAddKeyword, btnEditKeyword, btnDeleteKeyword, btnBack;
    private LinearLayout layoutKeywordList;

    private boolean isEditMode = false; // 편집 모드 활성화 여부

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_keyword);

        // View 초기화
        etKeyword = findViewById(R.id.etKeyword);
        btnAddKeyword = findViewById(R.id.btnAddKeyword);
        btnEditKeyword = findViewById(R.id.btnEditKeyword);
        btnDeleteKeyword = findViewById(R.id.btnDeleteKeyword);
        btnBack = findViewById(R.id.btnBack);
        layoutKeywordList = findViewById(R.id.layout_keyword_list);

        // 1) 뒤로가기 버튼
        btnBack.setOnClickListener(v -> {
            finish(); // 현재 액티비티 종료 -> 이전 화면으로 돌아감
        });

        // 2) 추가 버튼
        btnAddKeyword.setOnClickListener(v -> {
            String keyword = etKeyword.getText().toString().trim();
            if (!keyword.isEmpty()) {
                addKeywordItem(keyword);
                etKeyword.setText(""); // 입력창 비우기
            }
        });

        // 3) 편집 버튼
        btnEditKeyword.setOnClickListener(v -> {
            isEditMode = !isEditMode;
            updateEditMode();
        });

        // 4) 삭제 버튼
        btnDeleteKeyword.setOnClickListener(v -> {
            deleteSelectedKeywords();
        });
    }

    private void addKeywordItem(String keyword) {
        // 키워드 하나를 담을 레이아웃
        LinearLayout keywordLayout = new LinearLayout(this);
        keywordLayout.setOrientation(LinearLayout.HORIZONTAL);
        keywordLayout.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));
        keywordLayout.setPadding(8, 8, 8, 8);

        // 키워드 TextView
        TextView keywordTextView = new TextView(this);
        keywordTextView.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                7f
        ));
        keywordTextView.setText(keyword);
        keywordTextView.setTextSize(16f);

        // 키워드 CheckBox (초기에는 숨김)
        CheckBox keywordCheckBox = new CheckBox(this);
        keywordCheckBox.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                2f
        ));
        keywordCheckBox.setVisibility(View.GONE); // 처음에는 안 보이게
        keywordCheckBox.setTag("checkbox"); // 나중에 구분하기 쉽게 태그 달아둠

        // 레이아웃에 추가
        keywordLayout.addView(keywordTextView);
        keywordLayout.addView(keywordCheckBox);

        // 리스트 레이아웃에 추가
        layoutKeywordList.addView(keywordLayout);

        // 구분선 추가
        View divider = new View(this);
        LinearLayout.LayoutParams dividerParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                2
        );
        dividerParams.setMargins(0, 8, 0, 8);
        divider.setLayoutParams(dividerParams);
        divider.setBackgroundColor(0xFFCCCCCC); // 연한 회색

        layoutKeywordList.addView(divider);
    }

    private void updateEditMode() {
        int childCount = layoutKeywordList.getChildCount();
        for (int i = 0; i < childCount; i++) {
            View view = layoutKeywordList.getChildAt(i);
            if (view instanceof LinearLayout) {
                LinearLayout itemLayout = (LinearLayout) view;
                for (int j = 0; j < itemLayout.getChildCount(); j++) {
                    View child = itemLayout.getChildAt(j);
                    if (child instanceof CheckBox) {
                        child.setVisibility(isEditMode ? View.VISIBLE : View.GONE);
                    }
                }
            }
        }
        btnDeleteKeyword.setVisibility(isEditMode ? View.VISIBLE : View.GONE);
    }

    private void deleteSelectedKeywords() {
        int i = 0;
        while (i < layoutKeywordList.getChildCount()) {
            View view = layoutKeywordList.getChildAt(i);
            if (view instanceof LinearLayout) {
                LinearLayout itemLayout = (LinearLayout) view;
                CheckBox checkBox = null;
                for (int j = 0; j < itemLayout.getChildCount(); j++) {
                    View child = itemLayout.getChildAt(j);
                    if (child instanceof CheckBox) {
                        checkBox = (CheckBox) child;
                        break;
                    }
                }
                if (checkBox != null && checkBox.isChecked()) {
                    // 선택된 항목과 다음 divider까지 삭제
                    layoutKeywordList.removeViewAt(i); // 항목 삭제
                    if (i < layoutKeywordList.getChildCount()) {
                        View divider = layoutKeywordList.getChildAt(i);
                        if (divider instanceof View) {
                            layoutKeywordList.removeViewAt(i); // 구분선도 삭제
                        }
                    }
                    continue; // 삭제했으니 i는 그대로 (다음 것도 같은 위치에 있음)
                }
            }
            i++;
        }
    }
}

