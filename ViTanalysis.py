"""Export CIFAR-10 errors using Pillow only; rows=true, columns=predicted."""
import csv
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def export_analysis(dataset, predictions, confidences, checkpoint, mean_loss, epoch):
    classes = dataset.classes
    n = len(classes)
    assert len(predictions) == len(confidences) == len(dataset.targets)
    matrix = [[0] * n for _ in range(n)]
    errors = []
    for i, (actual, predicted, confidence) in enumerate(zip(dataset.targets, predictions, confidences)):
        matrix[actual][predicted] += 1
        if actual != predicted:
            errors.append(dict(index=i, actual=classes[actual], predicted=classes[predicted], confidence=confidence))
    out = Path(checkpoint).resolve().parent / 'analysis' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    def csv_write(name, header, rows):
        with (out / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
    csv_write('confusion_matrix.csv', ['actual / predicted'] + classes,
              [[classes[i]] + row for i, row in enumerate(matrix)])
    metrics = []
    for i, name in enumerate(classes):
        support = sum(matrix[i]); predicted = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append([name, support, tp, support-tp, recall, precision, f1])
    csv_write('per_class_metrics.csv', ['class','samples','correct','errors','accuracy_recall','precision','f1'], metrics)
    csv_write('errors.csv', ['test_index','actual','predicted','predicted_probability'],
              [[e['index'],e['actual'],e['predicted'],e['confidence']] for e in errors])
    images_dir = out / 'errors'; images_dir.mkdir()
    for e in errors:
        Image.fromarray(dataset.data[e['index']]).save(images_dir / f"{e['index']:05d}_true-{e['actual']}_pred-{e['predicted']}.png")
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    except OSError:
        font = small = ImageFont.load_default()
    # Each cell reports count and row percentage; diagonal shaded by recall, off-diagonal by error rate.
    canvas = Image.new('RGB',(1100,900),'white'); d=ImageDraw.Draw(canvas)
    d.text((25,15),'CIFAR-10 confusion matrix | rows: actual, columns: predicted',fill='black',font=font)
    for i,name in enumerate(classes):
        d.text((180+i*88,55),name,fill='black',font=small)
        d.text((20,98+i*76),name,fill='black',font=font)
        total=sum(matrix[i])
        for j in range(n):
            value=matrix[i][j]; fraction=value/total if total else 0
            intensity=int(150*fraction)
            color=(230-intensity,245-intensity//2,255) if i==j else (255,245-intensity,245-intensity)
            x,y=175+j*88,85+i*76
            d.rectangle((x,y,x+86,y+74),fill=color,outline='lightgray')
            d.text((x+8,y+12),str(value),fill='black',font=font)
            d.text((x+8,y+38),f'{fraction:.1%}',fill='black',font=small)
    canvas.save(out/'confusion_matrix.png')
    # Highest-confidence mistakes, up to 40; probability is a model score, not calibrated certainty.
    selected=sorted(errors,key=lambda e:e['confidence'],reverse=True)[:40]
    sheet=Image.new('RGB',(1000,max(1,(len(selected)+4)//5)*180+45),'white'); d=ImageDraw.Draw(sheet)
    d.text((12,8),'Highest-confidence errors (up to 40) | T: true, P: predicted',fill='black',font=font)
    for k,e in enumerate(selected):
        x,y=(k%5)*200,(k//5)*180+40
        im=Image.fromarray(dataset.data[e['index']]).resize((112,112),Image.Resampling.NEAREST)
        sheet.paste(im,(x+35,y))
        d.text((x+5,y+114),f"#{e['index']} T:{e['actual']}",fill='black',font=small)
        d.text((x+5,y+135),f"P:{e['predicted']} {e['confidence']:.1%}",fill='black',font=small)
    sheet.save(out/'error_examples.png')
    pairs=sorted([(matrix[i][j],classes[i],classes[j]) for i in range(n) for j in range(n) if i!=j],reverse=True)
    summary=dict(checkpoint=str(Path(checkpoint).resolve()),epoch=epoch,samples=len(predictions),
                 correct=len(predictions)-len(errors),errors=len(errors),accuracy=1-len(errors)/len(predictions),
                 mean_loss=mean_loss,macro_f1=sum(row[-1] for row in metrics)/n,
                 top_confusions=[dict(count=count,actual=a,predicted=b) for count,a,b in pairs[:10] if count])
    (out/'test_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# CIFAR-10 测试错误分析','',f"测试准确率：{summary['accuracy']:.2%}；错误：{len(errors)}/{len(predictions)}；平均loss：{mean_loss:.4f}。",
           f'权重：{checkpoint}；训练轮次：{epoch}。','',
           '混淆矩阵行是真实类别，列是预测类别；格内百分比按真实类别样本数计算。各类别准确率即该类别召回率。','',
           '|类别|正确/样本|准确率|Precision|F1|','|---|---:|---:|---:|---:|']
    for name,support,tp,err,rec,prec,f1 in metrics:
        lines.append(f'|{name}|{tp}/{support}|{rec:.2%}|{prec:.2%}|{f1:.4f}|')
    lines+=['','## 最多见的误判方向','']
    lines += [f'- {a} → {b}：{count}张' for count,a,b in pairs[:10] if count]
    lines+=['','![混淆矩阵](confusion_matrix.png)','','![错误示例](error_examples.png)','',
            'errors/包含全部错误原图；errors.csv保留官方测试集的0起始索引和预测概率。示例图按预测概率降序选取最多40张，不代表所有错误的分布；概率不是经过校准的置信保证。',
            '本分析用于描述固定模型的测试错误，不依据这些测试图片反复选择超参数。']
    (out/'错误分析.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Analysis output:',out,flush=True)
    return out
