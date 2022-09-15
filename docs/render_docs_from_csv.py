from pydoc import allmethods
import pandas as pd
import pdb
import math
from mdutils.mdutils import MdUtils
import numpy as np
import os


def check_nan(item):
    if (type(item) == float or type(item) == np.float64 or type(item) == np.float32) and math.isnan(item):
        return True
    else:
        return False

def title_by_cls(cls_str, cn):
    if not cn:
        ret = f'Weekly Classified Neural Radiance Fields - {cls_str} ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)'
    else:
        ret = f'每周分类神经辐射场 - {cls_str} ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)'
    return ret

def write_item_to_md(md, data, idx, cn=False):
    if cn:
        cur_title = data['title_cn'][idx]
    else:
        cur_title = data['title'][idx]
    cur_publisher = data['publisher'][idx]
    if not check_nan(cur_publisher):
        cur_title = cur_title + ", " + cur_publisher
    cur_link = data['link'][idx]
    md.write('  - ' + md.new_inline_link(link=cur_link, text=cur_title) + " | ")
    cur_code = data['code'][idx]
    if not check_nan(cur_code):
        md.write(md.new_inline_link(link=cur_code, text='[code]', bold_italics_code='cbi'))
        md.write("\n")
    else:
        md.write("[code]\n")
    if cn:
        cur_abstract = data['abstract_cn'][idx]
    else:
        cur_abstract = data['abstract'][idx]
    md.write("    > " + cur_abstract + "\n")
    return

def render_main_doc(meta_data_path="docs/weekly_nerf_meta_data.xlsx", cn=False):
    excel_data = pd.read_excel(meta_data_path)
    # Read the values of the file in the dataframe
    classes = ['lighting',	'editing', 'fast',	'dynamic',	'generalization', 'reconstruction',	'pose-slam', 'texture',	'semantic',	'human', 'video', 'others']
    data = pd.DataFrame(excel_data, columns=['week', 'title', 'publisher', 'abstract', 'link', 'code', 'title_cn', 'abstract_cn'] + classes)
    classified_path = "docs/classified_weekly_nerf"
    os.makedirs(classified_path, exist_ok=True)
    classified_path_cn = "docs/classified_weekly_nerf_cn"
    os.makedirs(classified_path_cn, exist_ok=True)
    if not cn:
        general_title = 'Weekly Classified Neural Radiance Fields ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)'
    else:
        general_title = '每周分类神经辐射场 ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)'
    all_md_fn = 'docs/weekly_nerf' if not cn else 'docs/weekly_nerf_cn'
    all_md = MdUtils(file_name=all_md_fn, title=general_title)
    if not cn:
        cls_md = {one_cls: MdUtils(file_name=os.path.join('docs/classified_weekly_nerf', one_cls),title=title_by_cls(one_cls, cn)) for one_cls in classes}
        cls_head = "## Filter by classes: \n [all](./weekly_nerf.md) | [dynamic](./classified_weekly_nerf/dynamic.md) | [editing](./classified_weekly_nerf/editing.md) | [fast](./classified_weekly_nerf/fast.md) | [generalization](./classified_weekly_nerf/generalization.md) | [human](./classified_weekly_nerf/human.md) | [video](./classified_weekly_nerf/video.md) | "
        cls_head2 = "[lighting](./classified_weekly_nerf/lighting.md) | [reconstruction](./classified_weekly_nerf/reconstruction.md) | [texture](./classified_weekly_nerf/texture.md) | [semantic](./classified_weekly_nerf/semantic.md) | [pose-slam](./classified_weekly_nerf/pose-slam.md) | [others](./classified_weekly_nerf/others.md) \n"
    else:
        cls_md = {one_cls: MdUtils(file_name=os.path.join('docs/classified_weekly_nerf_cn', one_cls),title=title_by_cls(one_cls, cn)) for one_cls in classes}
        all_md.write("\n NeRF研究QQ大群（300+成员）：706949479 \n")
        cls_head = "## 按类别筛选: \n [全部](./weekly_nerf_cn.md) | [动态](./classified_weekly_nerf_cn/dynamic.md) | [编辑](./classified_weekly_nerf_cn/editing.md) | [快速](./classified_weekly_nerf_cn/fast.md) | [泛化](./classified_weekly_nerf_cn/generalization.md) | [人体](./classified_weekly_nerf_cn/human.md) | [视频](./classified_weekly_nerf_cn/video.md) | "
        cls_head2 = "[光照](./classified_weekly_nerf_cn/lighting.md) | [重建](./classified_weekly_nerf_cn/reconstruction.md) | [纹理](./classified_weekly_nerf_cn/texture.md) | [语义](./classified_weekly_nerf_cn/semantic.md) | [姿态-SLAM](./classified_weekly_nerf_cn/pose-slam.md) | [其他](./classified_weekly_nerf_cn/others.md) \n"

    all_md.write(cls_head)
    all_md.write(cls_head2)

    # if cn:
        # all_md.write("## 大部分为机器翻译，少数论文手动翻译，有翻译错误可以PR修复。\n")
    # note this is different from above because the following is used in classified NeRFs
    if not cn:
        cls_head = "## Filter by classes: \n [all](../weekly_nerf.md) | [dynamic](./dynamic.md) | [editing](./editing.md) | [fast](./fast.md) | [generalization](./generalization.md) | [human](./human.md) | [video](./video.md) | "
        cls_head2 = "[lighting](./lighting.md) | [reconstruction](./reconstruction.md) | [texture](./texture.md) | [semantic](./semantic.md) | [pose-slam](./pose-slam.md) | [others](./others.md) \n"
    else:
        cls_head = "## 按类别筛选: \n [全部](../weekly_nerf_cn.md) | [动态](./dynamic.md) | [编辑](./editing.md) | [快速](./fast.md) | [泛化](./generalization.md) | [人体](./human.md) | [视频](./video.md) | "
        cls_head2 = "[光照](./lighting.md) | [重建](./reconstruction.md) | [纹理](./texture.md) | [语义](./semantic.md) | [姿态-SLAM](./pose-slam.md) | [其他](./others.md) \n"
    for cls in cls_md:
        cls_md[cls].write(cls_head)
        cls_md[cls].write(cls_head2)
        # if cn:
        #     cls_md[cls].write("## 大部分为机器翻译，少数论文手动翻译，有翻译错误可以PR修复。\n")
    data_len = len(data['week'])
    week = ""
    for idx in range(data_len):
        print(f"Generating {idx} / {data_len} ...")
        cur_week = data['week'][idx]
        if cur_week != week:
            week = cur_week
            all_md.write("## " + week + "\n")
            for cls in cls_md:
                cls_md[cls].write("## " + week + "\n")
        write_item_to_md(all_md, data, idx, cn)
        for cls in classes:
            cur_cls = data[cls][idx]
            if not check_nan(cur_cls):
                write_item_to_md(cls_md[cls], data, idx, cn)
    # if not cn:
    #     all_md.write("## " + 'Old papers\n')
    #     all_md.write("Refer to the [awesome-NeRF code repo](https://github.com/yenchenlin/awesome-NeRF).\n")
    # else:
    #     all_md.write("## " + '旧论文\n')
    #     all_md.write("参考这个仓库： [awesome-NeRF](https://github.com/yenchenlin/awesome-NeRF).\n")
    all_md.create_md_file()
    for cls in cls_md:
        cls_md[cls].create_md_file()


if __name__ == '__main__':
    render_main_doc(cn=False)
    render_main_doc(cn=True)
