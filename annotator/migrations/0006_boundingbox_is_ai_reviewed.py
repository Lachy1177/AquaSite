from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotator', '0005_analysisimage_cluster_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='boundingbox',
            name='is_ai_reviewed',
            field=models.BooleanField(
                default=False,
                help_text='True once a human has explicitly confirmed or corrected the DINOv3 suggestion in the review queue.',
            ),
        ),
    ]
